from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.retrieval_utils import (
    _build_retrieval_query,
    _compute_dynamic_candidates,
    _is_short_query,
    _is_long_query,
    _is_standalone_query,
    _remove_stopwords,
)


class TestBuildRetrievalQuery:
    def test_disabled_returns_input(self):
        result = _build_retrieval_query("current question", [], enabled=False)
        assert result == "current question"

    def test_empty_history_returns_input(self):
        result = _build_retrieval_query("current question", [], enabled=True)
        assert result == "current question"

    def test_with_human_history(self):
        history = [
            HumanMessage(content="previous question"),
            AIMessage(content="some answer"),
            HumanMessage(content="earlier question"),
        ]
        result = _build_retrieval_query("current question", history, max_user_messages=2)
        # reversed order: most recent human message first
        assert result == "current question earlier question previous question"

    def test_ignores_non_human_messages(self):
        history = [
            SystemMessage(content="system msg"),
            AIMessage(content="assistant msg"),
            HumanMessage(content="human msg"),
        ]
        result = _build_retrieval_query("current", history, max_user_messages=2)
        assert result == "current human msg"

    def test_respects_max_chars(self):
        history = [HumanMessage(content="a very long previous question with lots of words")]
        result = _build_retrieval_query(
            "current", history, max_user_messages=1, max_chars=20
        )
        assert len(result) <= 20
        assert result.startswith("current")

    def test_deduplicates_current_input(self):
        history = [HumanMessage(content="current")]
        result = _build_retrieval_query("current", history, max_user_messages=1)
        assert result == "current"

    def test_normalizes_whitespace(self):
        history = [HumanMessage(content="  spaced   out  ")]
        result = _build_retrieval_query("current", history, max_user_messages=1)
        assert result == "current spaced out"

    def test_strips_conversational_noise(self):
        history = [
            HumanMessage(content="ok danke, was ist ddos?"),
            AIMessage(content="some answer"),
            HumanMessage(content="hallo, guten tag!"),
        ]
        result = _build_retrieval_query("hallo, was sind grundlagen?", history, max_user_messages=2)
        # Standalone definition query: only the cleaned current input, no history mixed in
        assert result == "was sind grundlagen"

    def test_purely_conversational_returns_empty(self):
        history = [HumanMessage(content="was ist ddos")]
        result = _build_retrieval_query("ok danke!", history, max_user_messages=1)
        assert result == ""

    def test_german_confirmation_returns_empty(self):
        history = [HumanMessage(content="was ist ddos")]
        assert _build_retrieval_query("ja, super!", history) == ""
        assert _build_retrieval_query("alles klar!!!", history) == ""
        assert _build_retrieval_query("perfekt", history) == ""
        assert _build_retrieval_query("stimmt, genau", history) == ""

    def test_english_confirmation_returns_empty(self):
        history = [HumanMessage(content="was ist ddos")]
        assert _build_retrieval_query("yes, great!", history) == ""
        assert _build_retrieval_query("awesome, thanks!", history) == ""
        assert _build_retrieval_query("sounds good", history) == ""
        assert _build_retrieval_query("got it, roger", history) == ""

    def test_emoji_returns_empty(self):
        history = [HumanMessage(content="was ist ddos")]
        assert _build_retrieval_query("👋", history) == ""
        assert _build_retrieval_query("🙏🏻", history) == ""
        assert _build_retrieval_query("👍🏿", history) == ""
        assert _build_retrieval_query("✌️", history) == ""
        assert _build_retrieval_query("🙂", history) == ""
        assert _build_retrieval_query("🎉", history) == ""

    def test_regional_greeting_returns_empty(self):
        history = [HumanMessage(content="was ist ddos")]
        assert _build_retrieval_query("moin!", history) == ""
        assert _build_retrieval_query("tach", history) == ""
        assert _build_retrieval_query("howdy", history) == ""

    def test_farewell_returns_empty(self):
        history = [HumanMessage(content="was ist ddos")]
        assert _build_retrieval_query("bis dann", history) == ""
        assert _build_retrieval_query("see ya!", history) == ""
        assert _build_retrieval_query("have a good one", history) == ""
        assert _build_retrieval_query("ciao", history) == ""

    def test_wellbeing_returns_empty(self):
        history = [HumanMessage(content="was ist ddos")]
        assert _build_retrieval_query("wie geht's dir?", history) == ""
        assert _build_retrieval_query("how are you", history) == ""
        assert _build_retrieval_query("what's up", history) == ""

    def test_thanks_variants_returns_empty(self):
        history = [HumanMessage(content="was ist ddos")]
        assert _build_retrieval_query("thanks a lot", history) == ""
        assert _build_retrieval_query("vielen herzlichen dank", history) == ""
        assert _build_retrieval_query("thx", history) == ""
        assert _build_retrieval_query("ty", history) == ""

    def test_apology_returns_empty(self):
        history = [HumanMessage(content="was ist ddos")]
        assert _build_retrieval_query("sorry", history) == ""
        assert _build_retrieval_query("entschuldigung", history) == ""
        assert _build_retrieval_query("my bad", history) == ""

    def test_mixed_conversational_and_semantic_not_empty(self):
        # Conversational noise + standalone query → standalone query survives
        history = [HumanMessage(content="was ist ddos")]
        result = _build_retrieval_query("ja, was ist ddos?", history)
        assert result == "was ist ddos"

        result = _build_retrieval_query("thanks, explain ransomware", history)
        assert result == "explain ransomware"

    def test_case_insensitive_noise(self):
        history = [HumanMessage(content="was ist ddos")]
        assert _build_retrieval_query("SUPER!!!", history) == ""
        assert _build_retrieval_query("ThAnKs", history) == ""
        assert _build_retrieval_query("MoIn", history) == ""


class TestComputeDynamicCandidates:
    def test_short_query(self):
        vector, keyword = _compute_dynamic_candidates("what is", base_vector=40, base_keyword=40)
        assert vector < 40
        assert keyword >= 40

    def test_normal_query(self):
        vector, keyword = _compute_dynamic_candidates(
            "how does the incident response work", base_vector=40, base_keyword=40
        )
        assert vector == 40
        assert keyword == 40

    def test_long_query(self):
        vector, keyword = _compute_dynamic_candidates(
            "explain in great detail how the BSI incident response procedure works step by step for every organization in cybersecurity",
            base_vector=40,
            base_keyword=40,
        )
        assert vector > 40
        assert keyword < 40

    def test_disabled_returns_base(self):
        vector, keyword = _compute_dynamic_candidates(
            "what is", base_vector=40, base_keyword=40, enabled=False
        )
        assert vector == 40
        assert keyword == 40

    def test_short_query_threshold(self):
        assert _is_short_query("what is this") is True
        assert _is_short_query("what is this thing here now") is False

    def test_long_query_threshold(self):
        assert _is_long_query("explain in detail how the BSI incident response procedure works step by step for every organization") is True
        assert _is_long_query("how does this work") is False


class TestIsStandaloneQuery:
    def test_definition_patterns(self):
        assert _is_standalone_query("was ist ddos") is True
        assert _is_standalone_query("was bedeutet phishing") is True
        assert _is_standalone_query("was sind firewalls") is True
        assert _is_standalone_query("erkläre ransomware") is True
        assert _is_standalone_query("definition von malware") is True
        assert _is_standalone_query("mehr infos zu agentix") is True

    def test_english_definition_patterns(self):
        assert _is_standalone_query("what is ddos") is True
        assert _is_standalone_query("what does phishing mean") is True
        assert _is_standalone_query("explain ransomware") is True
        assert _is_standalone_query("definition of malware") is True

    def test_context_referencing_not_standalone(self):
        # Queries that reference prior context via pronouns should NOT be standalone
        assert _is_standalone_query("was ist das") is False
        assert _is_standalone_query("erkläre mir das bitte") is False
        assert _is_standalone_query("was bedeutet es") is False
        assert _is_standalone_query("was ist dies") is False
        assert _is_standalone_query("what is it") is False
        assert _is_standalone_query("explain that") is False

    def test_short_non_definition_not_standalone(self):
        # Short queries without a clear definition pattern are NOT standalone,
        # because they could be follow-ups to a prior topic.
        assert _is_standalone_query("phishing erklärung") is False
        assert _is_standalone_query("ddos info") is False

    def test_follow_up_question_not_standalone(self):
        # Longer follow-up questions with context references
        assert _is_standalone_query("wie funktioniert das im detail") is False
        assert _is_standalone_query("und wie verhält es sich damit") is False


class TestRemoveStopwords:
    def test_removes_german_stopwords(self):
        assert _remove_stopwords("was ist ddos") == "ddos"
        # "infos" is also a stopword, so only "agentix" remains
        assert _remove_stopwords("mehr infos zu agentix bitte") == "agentix"

    def test_removes_english_stopwords(self):
        assert _remove_stopwords("what is ddos") == "ddos"
        # "info" and "about" are also stopwords, so only "phishing" remains
        assert _remove_stopwords("more info about phishing") == "phishing"

    def test_keeps_original_when_all_stopwords(self):
        # When all words are stopwords, keep the original to avoid empty queries
        assert _remove_stopwords("was ist das") == "was ist das"

    def test_empty_input(self):
        assert _remove_stopwords("") == ""
        assert _remove_stopwords("   ") == ""


class TestStandaloneHistorySkipping:
    def test_standalone_query_skips_history(self):
        history = [
            HumanMessage(content="was ist agentix"),
            AIMessage(content="some answer"),
            HumanMessage(content="mehr infos zu agentix"),
        ]
        result = _build_retrieval_query("was ist ddos", history, max_user_messages=2)
        # Standalone definition query should NOT be mixed with history
        assert result == "was ist ddos"

    def test_standalone_detection_disabled_uses_history(self):
        history = [
            HumanMessage(content="was ist agentix"),
        ]
        result = _build_retrieval_query(
            "was ist ddos", history, max_user_messages=1, standalone_detection=False
        )
        assert "was ist ddos" in result
        assert "was ist agentix" in result

    def test_non_standalone_uses_history(self):
        history = [HumanMessage(content="was ist ddos")]
        result = _build_retrieval_query("wie funktioniert das", history, max_user_messages=1)
        assert "wie funktioniert das" in result
        assert "was ist ddos" in result
