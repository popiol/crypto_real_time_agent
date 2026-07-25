"""Shared structured-diff helpers for LLM-driven code generation.

Used by any pipeline step that generates Python code via LLM and needs
incremental fix attempts using line-level diffs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CodeChange(BaseModel):
    action: Literal["add", "remove"] = Field(
        description="'add' inserts code after the given line; 'remove' deletes it."
    )
    line: int = Field(
        description=(
            "1-indexed line number. For 'add', new code is inserted after this line "
            "(0 = before line 1). For 'remove', this line is deleted."
        )
    )
    code: str = Field(
        default="",
        description="Source text to insert. Only used for 'add'. Use \\n for multiple lines.",
    )


class CodeDiff(BaseModel):
    changes: list[CodeChange] = Field(
        description="Ordered list of line-level changes that together fix the syntax error."
    )


def apply_changes(code: str, changes: list[CodeChange]) -> str:
    """Apply a list of line-level changes to code, returning the updated source."""
    lines = code.splitlines()
    for change in sorted(changes, key=lambda c: (c.line, c.action == "add"), reverse=True):
        if change.action == "remove":
            idx = change.line - 1
            if 0 <= idx < len(lines):
                lines.pop(idx)
        elif change.action == "add":
            new_lines = change.code.splitlines() if change.code else []
            lines[change.line:change.line] = new_lines
    return "\n".join(lines)
