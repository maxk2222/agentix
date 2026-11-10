from langchain.schema import Document

from agent.processing import clean_documents, chunk_documents


def test_clean_documents_removes_disallowed_control_characters() -> None:
    docs = [
        Document(
            page_content="Aus ungeprü\x00en\x01 Backups\nSchadso\x08ware\tprüfen\x7f",
            metadata={"source": "checkliste.pdf"},
        )
    ]

    cleaned = clean_documents(docs)

    assert len(cleaned) == 1
    assert "\x00" not in cleaned[0].page_content
    assert "\x01" not in cleaned[0].page_content
    assert "\x08" not in cleaned[0].page_content
    assert "\x7f" not in cleaned[0].page_content
    assert cleaned[0].page_content == "Aus ungeprüen Backups\nSchadsoware prüfen"
    assert cleaned[0].metadata == {"source": "checkliste.pdf"}


def test_clean_documents_normalizes_common_ligatures() -> None:
    docs = [Document(page_content="Auﬀällige ﬁnale ﬂache Aﬃnität und Aﬄikt")]

    cleaned = clean_documents(docs)

    assert cleaned[0].page_content == "Auffällige finale flache Affinität und Afflikt"


def test_clean_documents_preserves_structural_newlines() -> None:
    docs = [Document(page_content="Erste Zeile\n\nZweite\tZeile")]

    cleaned = clean_documents(docs)

    assert cleaned[0].page_content == "Erste Zeile\nZweite Zeile"


def test_chunk_documents_respects_character_chunk_size() -> None:
    long_text = "A" * 2500
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

    splits = chunk_documents(docs, chunk_size=1000, chunk_overlap=100)

    assert len(splits) >= 3
    for split in splits:
        assert len(split.page_content) <= 1000


def test_chunk_documents_overlap_in_characters() -> None:
    text = "word " * 400
    docs = [Document(page_content=text, metadata={"source": "test.txt"})]

    splits = chunk_documents(docs, chunk_size=500, chunk_overlap=50)

    assert len(splits) >= 2
    for split in splits:
        assert len(split.page_content) <= 500
