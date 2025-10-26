"""Tests for callback system."""

from unittest.mock import AsyncMock

import pytest
from conftest import make_mock_response

from udspy import BaseCallback, Predict, settings
from udspy.signature import InputField, OutputField, Signature
from udspy.tool import Tool


class CallbackRecorder(BaseCallback):
    """Test callback that records all events."""

    def __init__(self):
        self.events = []

    def on_module_start(self, call_id, instance, inputs):
        self.events.append(("module_start", call_id, type(instance).__name__, inputs))

    def on_module_end(self, call_id, outputs, exception):
        self.events.append(("module_end", call_id, outputs, exception))

    def on_lm_start(self, call_id, instance, inputs):
        self.events.append(("lm_start", call_id, type(instance).__name__, inputs))

    def on_lm_end(self, call_id, outputs, exception):
        self.events.append(("lm_end", call_id, outputs, exception))

    def on_tool_start(self, call_id, instance, inputs):
        self.events.append(("tool_start", call_id, instance.name, inputs))

    def on_tool_end(self, call_id, outputs, exception):
        self.events.append(("tool_end", call_id, outputs, exception))


class QA(Signature):
    """Simple QA signature for testing."""

    question: str = InputField()
    answer: str = OutputField()


@pytest.mark.asyncio
async def test_module_callbacks():
    """Test that module callbacks are invoked."""
    callback = CallbackRecorder()
    predictor = Predict(QA, callbacks=[callback])

    # Mock the OpenAI API response
    mock_response = make_mock_response("[[ ## answer ## ]]\n4")
    settings.aclient.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await predictor.aforward(question="What is 2+2?")

    assert result.answer == "4"
    # Check module callbacks were called
    module_starts = [e for e in callback.events if e[0] == "module_start"]
    module_ends = [e for e in callback.events if e[0] == "module_end"]

    assert len(module_starts) == 1
    assert len(module_ends) == 1
    assert module_starts[0][2] == "Predict"  # instance type
    # The inputs dict contains both module params and user inputs
    inputs_dict = module_starts[0][3]
    assert "inputs" in inputs_dict
    assert inputs_dict["inputs"]["question"] == "What is 2+2?"


@pytest.mark.asyncio
async def test_lm_callbacks():
    """Test that LM callbacks are invoked."""
    callback = CallbackRecorder()
    predictor = Predict(QA, callbacks=[callback])

    # Mock the OpenAI API response
    mock_response = make_mock_response("[[ ## answer ## ]]\nParis")
    settings.aclient.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await predictor.aforward(question="What is the capital of France?")

    assert result.answer == "Paris"
    # Check LM callbacks were called
    lm_starts = [e for e in callback.events if e[0] == "lm_start"]
    lm_ends = [e for e in callback.events if e[0] == "lm_end"]

    assert len(lm_starts) >= 1
    assert len(lm_ends) >= 1
    assert lm_starts[0][2] == "Predict"  # instance type


@pytest.mark.asyncio
async def test_tool_callbacks():
    """Test that tool callbacks are invoked."""

    def mock_tool(query: str) -> str:
        """Mock search tool."""
        return f"Results for: {query}"

    callback = CallbackRecorder()
    tool = Tool(mock_tool, name="search", description="Search tool", callbacks=[callback])

    result = await tool.acall(query="test query")

    assert result == "Results for: test query"
    # Check tool callbacks were called
    tool_starts = [e for e in callback.events if e[0] == "tool_start"]
    tool_ends = [e for e in callback.events if e[0] == "tool_end"]

    assert len(tool_starts) == 1
    assert len(tool_ends) == 1
    assert tool_starts[0][2] == "search"  # tool name
    # Tool acall passes kwargs, so inputs will have those
    assert "kwargs" in tool_starts[0][3] or "query" in tool_starts[0][3]
    # Check we captured the query argument somehow
    if "kwargs" in tool_starts[0][3]:
        assert tool_starts[0][3]["kwargs"]["query"] == "test query"
    else:
        assert tool_starts[0][3]["query"] == "test query"


@pytest.mark.asyncio
async def test_global_callbacks():
    """Test that global callbacks from settings are invoked."""
    callback = CallbackRecorder()

    # Mock the OpenAI API response
    mock_response = make_mock_response("[[ ## answer ## ]]\nTest answer")
    settings.aclient.chat.completions.create = AsyncMock(return_value=mock_response)

    # Save the mocked client
    mock_client = settings.aclient

    # Configure global callbacks (keep using the mocked client)
    with settings.context(callbacks=[callback], aclient=mock_client):
        predictor = Predict(QA)
        result = await predictor.aforward(question="Test question")

        assert result.answer == "Test answer"
        # Check callbacks were invoked
        assert len(callback.events) > 0
        module_starts = [e for e in callback.events if e[0] == "module_start"]
        assert len(module_starts) >= 1


@pytest.mark.asyncio
async def test_callback_exception_handling():
    """Test that exceptions in callbacks don't break execution."""

    class FaultyCallback(BaseCallback):
        def on_module_start(self, call_id, instance, inputs):
            raise ValueError("Callback error")

    good_callback = CallbackRecorder()
    predictor = Predict(QA, callbacks=[FaultyCallback(), good_callback])

    # Mock the OpenAI API response
    mock_response = make_mock_response("[[ ## answer ## ]]\n4")
    settings.aclient.chat.completions.create = AsyncMock(return_value=mock_response)

    # Should still work despite callback error
    result = await predictor.aforward(question="What is 2+2?")
    assert result.answer == "4"


def test_sync_callbacks():
    """Test that callbacks work with sync methods."""
    callback = CallbackRecorder()
    predictor = Predict(QA, callbacks=[callback])

    # Mock the OpenAI API response
    mock_response = make_mock_response("[[ ## answer ## ]]\nPython is a programming language")
    settings.aclient.chat.completions.create = AsyncMock(return_value=mock_response)

    result = predictor(question="What is Python?")

    assert result.answer == "Python is a programming language"
    # Check callbacks were invoked
    module_starts = [e for e in callback.events if e[0] == "module_start"]
    assert len(module_starts) >= 1
