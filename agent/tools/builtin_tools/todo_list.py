"""Run-local todo list tool, automatically available to every agent run.

The tool itself is stateless: the actual list lives in a ``TodoListState``
instance that the ``AgentRunner`` creates per run and passes via the tool
execution context.  This keeps lists isolated between runs and between
subagents, and discards them automatically when the run ends.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

TODO_TOOL_NAME = "agent_todo"
TODO_STATUSES = ("pending", "in_progress", "completed", "skipped")
# Statuses that mean "no work will happen on this task anymore".
OPEN_STATUSES = ("pending", "in_progress")

_DEFAULT_MAX_TASKS = 20
_MAX_TASKS_CAP = 50
_TITLE_LIMIT = 200
_NOTE_LIMIT = 1_000


class TodoListState:
    """Mutable, run-scoped todo list.

    ``version`` is bumped on every mutation so the runner can detect changes
    and stream fresh snapshots to the client.
    """

    def __init__(self) -> None:
        self._tasks: List[Dict[str, Any]] = []
        self._counter = 0
        self.version = 0

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        return list(self._tasks)

    def _resolve_id(self, task_id: Any) -> str:
        """Normalize task ids; accept ``t1``, ``1``, ``T1`` or int 1."""
        raw = str(task_id or "").strip().lower()
        if raw and raw[0] != "t":
            raw = f"t{raw}"
        for task in self._tasks:
            if task["id"] == raw:
                return task["id"]
        known = ", ".join(task["id"] for task in self._tasks) or "none"
        raise ValueError(f"Unknown task id '{task_id}'. Existing ids: {known}")

    def add(
        self,
        title: str,
        note: Optional[str] = None,
        status: str = "pending",
        max_tasks: int = _DEFAULT_MAX_TASKS,
    ) -> Dict[str, Any]:
        title = (title or "").strip()
        if not title:
            raise ValueError("A non-empty 'title' is required to add a task")
        if len(self._tasks) >= max_tasks:
            raise ValueError(f"Todo list is full ({max_tasks} tasks). Remove or complete tasks first.")
        if status not in TODO_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Allowed: {', '.join(TODO_STATUSES)}")
        self._counter += 1
        task = {
            "id": f"t{self._counter}",
            "title": title[:_TITLE_LIMIT],
            "status": status,
        }
        note = (note or "").strip()
        if note:
            task["note"] = note[:_NOTE_LIMIT]
        self._tasks.append(task)
        self.version += 1
        return task

    def update(
        self,
        task_id: str,
        title: Optional[str] = None,
        note: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        tid = self._resolve_id(task_id)
        task = next(t for t in self._tasks if t["id"] == tid)
        changed = False
        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("'title' must not be empty when provided")
            if task["title"] != title[:_TITLE_LIMIT]:
                task["title"] = title[:_TITLE_LIMIT]
                changed = True
        if note is not None:
            note = note.strip()
            new_note = note[:_NOTE_LIMIT] if note else None
            if task.get("note") != new_note:
                if new_note:
                    task["note"] = new_note
                else:
                    task.pop("note", None)
                changed = True
        if status is not None:
            if status not in TODO_STATUSES:
                raise ValueError(f"Invalid status '{status}'. Allowed: {', '.join(TODO_STATUSES)}")
            if task["status"] != status:
                task["status"] = status
                changed = True
        if changed:
            self.version += 1
        return task

    def remove(self, task_id: str) -> Dict[str, Any]:
        tid = self._resolve_id(task_id)
        task = next(t for t in self._tasks if t["id"] == tid)
        self._tasks = [t for t in self._tasks if t["id"] != tid]
        self.version += 1
        return task

    def clear(self) -> int:
        removed = len(self._tasks)
        self._tasks = []
        if removed:
            self.version += 1
        return removed

    def mark_unfinished_skipped(self) -> int:
        """Sweep every still-open task to ``skipped`` — used when the run ends.

        The run is over: leftover pending/in_progress tasks will never be
        worked on, and marking them explicitly lets the dashboard render them
        struck through instead of pretending work continues. One version
        bump for the whole sweep.
        """
        changed = 0
        for task in self._tasks:
            if task["status"] in OPEN_STATUSES:
                task["status"] = "skipped"
                changed += 1
        if changed:
            self.version += 1
        return changed

    def snapshot(self) -> Dict[str, Any]:
        return {"version": self.version, "tasks": [dict(t) for t in self._tasks]}

    def render(self) -> str:
        if not self._tasks:
            return "Todo list is empty."
        done = sum(1 for t in self._tasks if t["status"] == "completed")
        skipped = sum(1 for t in self._tasks if t["status"] == "skipped")
        open_count = len(self._tasks) - done - skipped
        summary = f"Todo list - {open_count} open, {done} done"
        if skipped:
            summary += f", {skipped} skipped"
        lines = [summary + ":"]
        for task in self._tasks:
            status = task["status"]
            marker = (
                "[x]" if status == "completed"
                else "[>]" if status == "in_progress"
                else "[-]" if status == "skipped"
                else "[ ]"
            )
            label = f"[{status}]" if status != "completed" else "[completed]"
            line = f"{marker} {task['id']} {label} {task['title']}"
            if task.get("note"):
                line += f" - {task['note']}"
            lines.append(line)
        return "\n".join(lines)


class AgentTodoTool:
    """Lets the agent maintain a scratch todo list for the current run."""

    name = TODO_TOOL_NAME

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Maintain a scratch todo list to organize multi-step work in this run. "
                    "Create tasks for non-trivial requests, keep one task in_progress at a time, "
                    "and mark tasks completed as you finish them. The list is temporary and "
                    "visible to the user while you work."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "update", "complete", "remove", "list", "clear"],
                            "description": (
                                "add: create a task (requires title); update: change title/note/status "
                                "(requires task_id); complete: mark task done (requires task_id); "
                                "remove: delete task (requires task_id); list: show all tasks; "
                                "clear: delete all tasks."
                            ),
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Id of the task to modify, e.g. 't1'. Required for update/complete/remove.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Short task title. Required for add; optional new title for update.",
                        },
                        "note": {
                            "type": "string",
                            "description": "Optional short detail for the task. Send an empty string in update to clear it.",
                        },
                        "status": {
                            "type": "string",
                            "enum": list(TODO_STATUSES),
                            "description": (
                                "New status for add/update. Defaults to 'pending' on add. "
                                "Use 'skipped' for tasks you deliberately drop."
                            ),
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    async def execute(
        self,
        action: str,
        task_id: Optional[str] = None,
        title: Optional[str] = None,
        note: Optional[str] = None,
        status: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = (context or {}).get("todo_state")
        if not isinstance(state, TodoListState):
            raise RuntimeError("Todo list is not available for this run")
        tool_config = (context or {}).get("tool_config") or {}
        try:
            max_tasks = max(1, min(_MAX_TASKS_CAP, int(tool_config.get("max_tasks") or _DEFAULT_MAX_TASKS)))
        except (TypeError, ValueError):
            max_tasks = _DEFAULT_MAX_TASKS

        action = (action or "").strip().lower()
        if action == "add":
            state.add(title or "", note=note, status=status or "pending", max_tasks=max_tasks)
        elif action == "update":
            if not task_id:
                raise ValueError("'task_id' is required for action 'update'")
            state.update(task_id, title=title, note=note, status=status)
        elif action == "complete":
            if not task_id:
                raise ValueError("'task_id' is required for action 'complete'")
            state.update(task_id, status="completed")
        elif action == "remove":
            if not task_id:
                raise ValueError("'task_id' is required for action 'remove'")
            state.remove(task_id)
        elif action == "list":
            pass
        elif action == "clear":
            state.clear()
        else:
            raise ValueError(
                f"Unknown action '{action}'. Allowed: add, update, complete, remove, list, clear"
            )

        return {"content": state.render()}
