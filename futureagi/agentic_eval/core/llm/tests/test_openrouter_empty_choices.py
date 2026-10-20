"""Regression tests for empty chat completion choices in OpenRouter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_eval.core.llm.openrouter import OpenRouter


def _openrouter_with_completion(completion):
    llm = OpenRouter.__new__(OpenRouter)
    llm.provider = "openrouter"
    llm.model_name = "test-model"
    llm.max_tokens = 16
    llm.temperature = 0
    create = Mock(return_value=completion)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return llm, create


def test_basic_request_raises_when_completion_has_no_choices():
    llm, create = _openrouter_with_completion(SimpleNamespace(choices=[]))

    with pytest.raises(ValueError, match="LLM provider returned no choices"):
        llm.basic_request("test prompt")

    create.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "test prompt"}],
        max_tokens=16,
        temperature=0,
    )


def test_basic_request_returns_content_from_first_choice():
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="response text"))]
    )
    llm, _ = _openrouter_with_completion(completion)

    assert llm.basic_request("test prompt") == "response text"
