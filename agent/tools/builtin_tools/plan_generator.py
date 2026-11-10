"""Plan generator builtin tool."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlanGeneratorTool:
    """Generate a structured markdown plan from a goal, context and constraints."""

    name = "plan_generator"
    description = (
        "Generate a structured plan for a given goal. "
        "Use 'goal' (required), optional 'context', 'constraints', "
        "'detail_level' ('brief', 'standard', 'detailed'), and 'include_risks' (default true)."
    )

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "The main goal or objective of the plan.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context or background information.",
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of constraints or limitations to consider.",
                        },
                        "detail_level": {
                            "type": "string",
                            "enum": ["brief", "standard", "detailed"],
                            "description": "Level of detail for the plan.",
                            "default": "standard",
                        },
                        "include_risks": {
                            "type": "boolean",
                            "description": "Include a risk assessment section.",
                            "default": True,
                        },
                    },
                    "required": ["goal"],
                },
            },
        }

    def execute(
        self,
        goal: str,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        detail_level: str = "standard",
        include_risks: bool = True,
    ) -> str:
        lines: List[str] = []
        lines.append(f"# Plan: {goal}")
        lines.append("")

        if context:
            lines.append("## Context")
            lines.append(context)
            lines.append("")

        lines.append("## Goal")
        lines.append(goal)
        lines.append("")

        if constraints:
            lines.append("## Constraints")
            for c in constraints:
                lines.append(f"- {c}")
            lines.append("")

        lines.append("## Steps")
        if detail_level == "brief":
            lines.append("1. Analyze requirements")
            lines.append("2. Design solution")
            lines.append("3. Implement")
            lines.append("4. Test and validate")
            lines.append("5. Deploy or deliver")
        elif detail_level == "standard":
            lines.append("1. **Analyze requirements** — Understand the goal, context, and constraints.")
            lines.append("2. **Research and gather data** — Collect relevant information and resources.")
            lines.append("3. **Design solution** — Create a detailed approach and break into sub-tasks.")
            lines.append("4. **Implement** — Execute the planned steps with regular checkpoints.")
            lines.append("5. **Test and validate** — Verify outcomes against the goal and constraints.")
            lines.append("6. **Iterate if needed** — Refine based on feedback or issues.")
            lines.append("7. **Deploy or deliver** — Finalize and hand over the result.")
        else:  # detailed
            lines.append("1. **Analyze requirements** — Break down the goal into specific, measurable objectives. Identify stakeholders and success criteria.")
            lines.append("2. **Research and gather data** — Collect all relevant background information, reference materials, and domain knowledge. Document assumptions.")
            lines.append("3. **Identify dependencies** — Map out internal and external dependencies, prerequisites, and blockers.")
            lines.append("4. **Design solution** — Develop a detailed technical or procedural approach. Create diagrams or outlines if applicable.")
            lines.append("5. **Create sub-tasks** — Break the solution into granular, actionable tasks with estimated effort and owners.")
            lines.append("6. **Implement** — Execute tasks in dependency order. Maintain a change log and version control.")
            lines.append("7. **Test and validate** — Perform unit, integration, and acceptance testing. Compare results against success criteria.")
            lines.append("8. **Iterate and refine** — Address issues, incorporate feedback, and improve the solution.")
            lines.append("9. **Document** — Write clear documentation for the outcome, decisions, and any manual steps.")
            lines.append("10. **Deploy or deliver** — Execute the final rollout, handover, or publication. Plan for post-delivery support.")
        lines.append("")

        if include_risks:
            lines.append("## Risks")
            lines.append("- Scope creep: requirements expanding beyond the original goal")
            lines.append("- Resource constraints: time, budget, or expertise limitations")
            lines.append("- Dependency delays: external blockers causing schedule slips")
            lines.append("- Quality issues: insufficient testing leading to defects")
            lines.append("- Communication gaps: misalignment between stakeholders")
            lines.append("")

        lines.append("## Acceptance Criteria")
        lines.append(f"- The goal '{goal}' is fully achieved.")
        lines.append("- All constraints are respected.")
        lines.append("- Output is tested and validated.")
        if detail_level == "detailed":
            lines.append("- Documentation is complete and reviewed.")
            lines.append("- Stakeholders have signed off on the result.")

        return "\n".join(lines)
