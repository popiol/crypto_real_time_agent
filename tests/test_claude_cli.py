"""Unit tests for the Claude CLI wrapper.

All tests mock subprocess.run — no actual claude binary required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.updater.claude_cli import _run, cli_call, cli_call_structured


class _Model(BaseModel):
    value: int
    label: str


def _proc(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    p = MagicMock()
    p.stdout = stdout
    p.returncode = returncode
    p.stderr = stderr
    return p


# ── cli_call ──────────────────────────────────────────────────────────────────

@patch("src.updater.claude_cli.subprocess.run")
def test_plain_text_returns_stdout(mock_run):
    mock_run.return_value = _proc("Hello world")
    assert cli_call(system="sys", user="prompt") == "Hello world"


@patch("src.updater.claude_cli.subprocess.run")
def test_system_prompt_flag_included(mock_run):
    mock_run.return_value = _proc("ok")
    cli_call(system="You are helpful", user="hi")
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("--system-prompt")
    assert cmd[idx + 1] == "You are helpful"


@patch("src.updater.claude_cli.subprocess.run")
def test_empty_system_omits_flag(mock_run):
    mock_run.return_value = _proc("ok")
    cli_call(system="", user="hi")
    cmd = mock_run.call_args[0][0]
    assert "--system-prompt" not in cmd


@patch("src.updater.claude_cli.subprocess.run")
def test_user_prompt_is_last_arg(mock_run):
    mock_run.return_value = _proc("ok")
    cli_call(system="", user="my question")
    cmd = mock_run.call_args[0][0]
    assert cmd[-1] == "my question"


@patch("src.updater.claude_cli.subprocess.run")
def test_bare_flag_not_present(mock_run):
    """--bare blocks OAuth auth; must never appear in the command."""
    mock_run.return_value = _proc("ok")
    cli_call(system="", user="hi")
    cmd = mock_run.call_args[0][0]
    assert "--bare" not in cmd


@patch("src.updater.claude_cli.subprocess.run")
def test_nonzero_exit_raises(mock_run):
    mock_run.return_value = _proc("", returncode=1, stderr="auth error")
    with pytest.raises(RuntimeError, match="claude CLI exited 1"):
        cli_call(system="", user="hi")


@patch("src.updater.claude_cli.subprocess.run")
def test_empty_output_raises(mock_run):
    mock_run.return_value = _proc("   ")
    with pytest.raises(RuntimeError, match="empty output"):
        cli_call(system="", user="hi")


@patch("src.updater.claude_cli.subprocess.run")
def test_none_stdout_raises(mock_run):
    mock_run.return_value = _proc(None)
    with pytest.raises(RuntimeError, match="empty output"):
        cli_call(system="", user="hi")


# ── cli_call_structured ───────────────────────────────────────────────────────

@patch("src.updater.claude_cli.subprocess.run")
def test_structured_parses_string_result(mock_run):
    payload = json.dumps({"value": 42, "label": "test"})
    envelope = json.dumps({"result": payload, "is_error": False})
    mock_run.return_value = _proc(envelope)
    result = cli_call_structured(system="", user="hi", output_type=_Model)
    assert result.value == 42
    assert result.label == "test"


@patch("src.updater.claude_cli.subprocess.run")
def test_structured_parses_dict_result(mock_run):
    envelope = json.dumps({"result": {"value": 7, "label": "x"}, "is_error": False})
    mock_run.return_value = _proc(envelope)
    result = cli_call_structured(system="", user="hi", output_type=_Model)
    assert result.value == 7
    assert result.label == "x"


@patch("src.updater.claude_cli.subprocess.run")
def test_structured_sends_json_schema_flag(mock_run):
    payload = json.dumps({"value": 1, "label": "a"})
    mock_run.return_value = _proc(json.dumps({"result": payload}))
    cli_call_structured(system="", user="hi", output_type=_Model)
    cmd = mock_run.call_args[0][0]
    assert "--json-schema" in cmd
    assert "--output-format" in cmd


@patch("src.updater.claude_cli.subprocess.run")
def test_structured_schema_matches_model(mock_run):
    payload = json.dumps({"value": 1, "label": "a"})
    mock_run.return_value = _proc(json.dumps({"result": payload}))
    cli_call_structured(system="", user="hi", output_type=_Model)
    cmd = mock_run.call_args[0][0]
    schema_idx = cmd.index("--json-schema")
    assert json.loads(cmd[schema_idx + 1]) == _Model.model_json_schema()


@patch("src.updater.claude_cli.subprocess.run")
def test_structured_invalid_json_raises(mock_run):
    mock_run.return_value = _proc(json.dumps({"result": "not json at all"}))
    with pytest.raises(Exception):
        cli_call_structured(system="", user="hi", output_type=_Model)


@patch("src.updater.claude_cli.subprocess.run")
def test_structured_wrong_schema_raises(mock_run):
    payload = json.dumps({"wrong_field": 99})
    mock_run.return_value = _proc(json.dumps({"result": payload}))
    with pytest.raises(Exception):
        cli_call_structured(system="", user="hi", output_type=_Model)
