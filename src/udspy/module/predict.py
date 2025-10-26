"""Predict module for LLM predictions based on signatures."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional

import regex as re
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from udspy.adapter import ChatAdapter
from udspy.callback import with_callbacks
from udspy.module.base import Module
from udspy.settings import settings
from udspy.signature import Signature
from udspy.streaming import Prediction, StreamChunk, StreamEvent

if TYPE_CHECKING:
    from udspy.history import History
    from udspy.tool import Tool

logger = logging.getLogger(__name__)


class Predict(Module):
    """Module for making LLM predictions based on a signature.

    This is an async-first module. The core method is `astream()` which yields
    StreamEvent objects. Use `aforward()` for async non-streaming, or `forward()`
    for sync usage.

    Example:
        ```python
        from udspy import Predict, Signature, InputField, OutputField

        class QA(Signature):
            '''Answer questions.'''
            question: str = InputField()
            answer: str = OutputField()

        predictor = Predict(QA)

        # Sync usage
        result = predictor(question="What is 2+2?")
        print(result.answer)

        # Async non-streaming
        result = await predictor.aforward(question="What is 2+2?")

        # Async streaming
        async for event in predictor.astream(question="What is 2+2?"):
            if isinstance(event, StreamChunk):
                print(event.delta, end="", flush=True)
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        *,
        model: str | None = None,
        tools: list["Tool"] | None = None,
        max_turns: int = 10,
        adapter: ChatAdapter | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize a Predict module.

        Args:
            signature: Signature defining inputs and outputs, or a string in
                      format "inputs -> outputs" (e.g., "question -> answer")
            model: Model name (overrides global default)
            tools: List of tool functions (decorated with @tool) or Pydantic models
            max_turns: Maximum number of LLM calls for tool execution loop (default: 10)
            adapter: Custom adapter (defaults to ChatAdapter)
            callbacks: Optional list of callback handlers for this module instance
            **kwargs: Additional arguments for chat completion (temperature, etc.)
        """
        super().__init__(callbacks=callbacks)
        from udspy.tool import Tool

        # Convert string signature to Signature class
        if isinstance(signature, str):
            signature = Signature.from_string(signature)

        self.signature = signature
        self.model = model or settings.default_model
        self.max_turns = max_turns
        self.adapter = adapter or ChatAdapter()
        self.kwargs = {**settings.default_kwargs, **kwargs}

        self.tool_callables: dict[str, Tool] = {}
        self.tool_schemas: list[Any] = []

        for tool in tools or []:
            self.tool_callables[tool.name] = tool
            self.tool_schemas.append(tool)

    @with_callbacks
    async def aexecute(
        self,
        *,
        stream: bool = False,
        auto_execute_tools: bool = True,
        history: Optional["History"] = None,
        **inputs: Any,
    ) -> Prediction:
        """Core execution method - handles both streaming and non-streaming.

        This is the single implementation point for LLM interaction. It always
        returns a Prediction, and emits events to the queue if one is active.

        Args:
            stream: If True, request streaming from OpenAI. If False, use regular API.
            auto_execute_tools: If True, automatically execute tools and continue.
                If False, return Prediction with tool_calls for manual handling.
            history: Optional History object for multi-turn conversations.
            **inputs: Input values matching the signature's input fields

        Returns:
            Final Prediction object (after all tool executions if auto_execute_tools=True)
        """
        from udspy.streaming import _stream_queue

        queue = _stream_queue.get()
        should_emit = queue is not None

        self._validate_inputs(inputs)
        messages = self._build_initial_messages(inputs, history)

        final_prediction = await self._execute_with_tools(
            messages,
            stream=stream,
            should_emit=should_emit,
            auto_execute_tools=auto_execute_tools,
            history=history,
        )

        if should_emit and queue:
            await queue.put(final_prediction)

        return final_prediction

    async def astream(
        self, *, auto_execute_tools: bool = True, history: Optional["History"] = None, **inputs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming method with optional automatic tool execution.

        Sets up streaming queue and yields events. Automatically handles multi-turn
        conversation when tools are present.

        Args:
            auto_execute_tools: If True, automatically execute tools and continue.
                If False, return Prediction with tool_calls for manual handling.
            history: Optional History object for multi-turn conversations.
            **inputs: Input values matching the signature's input fields

        Yields:
            StreamEvent objects (StreamChunk, Prediction, custom events)
        """
        async for event in super().astream(
            auto_execute_tools=auto_execute_tools, history=history, **inputs
        ):
            yield event

    def _validate_inputs(self, inputs: dict[str, Any]) -> None:
        """Validate that all required inputs are provided."""
        input_fields = self.signature.get_input_fields()
        for field_name in input_fields:
            if field_name not in inputs:
                raise ValueError(f"Missing required input field: {field_name}")

    def _build_initial_messages(
        self, inputs: dict[str, Any], history: Any = None
    ) -> list[dict[str, Any]]:
        """Build initial messages from inputs and optional history.

        Args:
            inputs: Input values from user
            history: Optional History object with existing conversation

        Returns:
            List of messages including history (if provided) and new user input
        """
        from udspy.history import History

        messages: list[dict[str, Any]] = []

        if history is not None:
            if isinstance(history, History):
                messages.extend(history.messages)
            else:
                raise TypeError(f"history must be a History object, got {type(history)}")

        if not messages or messages[0]["role"] != "system":
            messages.insert(
                0, {"role": "system", "content": self.adapter.format_instructions(self.signature)}
            )

        user_content = self.adapter.format_inputs(self.signature, inputs)
        user_msg = {"role": "user", "content": user_content}
        messages.append(user_msg)

        if history is not None and isinstance(history, History):
            history.messages.append(user_msg)

        return messages

    async def _execute_with_tools(
        self,
        messages: list[dict[str, Any]],
        stream: bool,
        should_emit: bool,
        auto_execute_tools: bool,
        history: Any = None,
    ) -> Prediction:
        """Execute multi-turn conversation with optional automatic tool execution.

        This is the core execution loop that handles both streaming and non-streaming.
        It always returns a final Prediction, and emits events if should_emit is True.

        Args:
            messages: Conversation messages
            stream: If True, request streaming from OpenAI
            should_emit: If True, emit events to active queue
            auto_execute_tools: If True, automatically execute tools. If False,
                return after first tool call.
            history: Optional History object to update with conversation

        Returns:
            Final Prediction object
        """
        final_prediction: Prediction | None = None

        for turn in range(self.max_turns):
            final_prediction = await self._execute_one_turn(
                messages, turn, stream=stream, should_emit=should_emit
            )

            if history is not None and final_prediction:
                self._update_history_with_prediction(history, final_prediction)

            if not (final_prediction and "tool_calls" in final_prediction):
                break

            if not auto_execute_tools:
                break

            if not self.tool_callables:
                break

            self._execute_tool_calls(messages, final_prediction.tool_calls, history)

        if turn >= self.max_turns - 1 and final_prediction and "tool_calls" in final_prediction:
            raise RuntimeError(f"Max turns ({self.max_turns}) reached without final answer")

        if final_prediction is None:
            raise RuntimeError("No prediction generated")

        return final_prediction

    async def _execute_one_turn(
        self, messages: list[dict[str, Any]], turn: int, stream: bool, should_emit: bool
    ) -> Prediction:
        """Execute one LLM turn (streaming or non-streaming).

        Args:
            messages: Conversation messages
            turn: Current turn number (0-indexed)
            stream: If True, request streaming from OpenAI
            should_emit: If True, emit events to active queue

        Returns:
            Prediction object for this turn
        """
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            **self.kwargs,
        }

        if turn == 0 and self.tool_schemas:
            tool_schemas = self.adapter.format_tool_schemas(self.tool_schemas)
            completion_kwargs["tools"] = tool_schemas

        if stream:
            return await self._process_streaming(completion_kwargs, should_emit)
        else:
            return await self._process_nonstreaming(completion_kwargs, should_emit)

    def _execute_tool_calls(
        self, messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]], history: Any = None
    ) -> None:
        """Execute tool calls and add results to messages.

        Args:
            messages: Conversation messages
            tool_calls: List of tool calls to execute
            history: Optional History object to update
        """
        import json

        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)

        if history is not None:
            from udspy.history import History

            if isinstance(history, History):
                history.messages.append(assistant_msg)

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = json.loads(tool_call["arguments"])

            if tool_name in self.tool_callables:
                try:
                    result = self.tool_callables[tool_name](**arguments)
                    content = str(result)
                except Exception as e:
                    content = f"Error executing tool: {e}"
            else:
                content = f"Error: Tool {tool_name} not found"

            tool_msg = {"role": "tool", "tool_call_id": tool_call["id"], "content": content}
            messages.append(tool_msg)

            if history is not None:
                from udspy.history import History

                if isinstance(history, History):
                    history.messages.append(tool_msg)

    def _update_history_with_prediction(self, history: Any, prediction: Prediction) -> None:
        """Update history with assistant's prediction.

        Args:
            history: History object to update
            prediction: Prediction from assistant
        """
        from udspy.history import History

        if not isinstance(history, History):
            return

        output_fields = self.signature.get_output_fields()
        content_parts = []

        for field_name in output_fields:
            if hasattr(prediction, field_name):
                value = getattr(prediction, field_name)
                if value:
                    content_parts.append(f"[[ ## {field_name} ## ]]\n{value}")

        content = "\n".join(content_parts) if content_parts else ""

        if hasattr(prediction, "tool_calls") and prediction.tool_calls:
            pass
        else:
            history.add_assistant_message(content)

    async def aforward(
        self, *, auto_execute_tools: bool = True, history: Any = None, **inputs: Any
    ) -> Prediction:
        """Async non-streaming method. Returns the final Prediction.

        Calls aexecute() with streaming disabled. If called from within a
        streaming context (i.e., another module is streaming), events will
        still be emitted to the active queue.

        When tools are used with auto_execute_tools=True (default), this returns
        the LAST prediction (after tool execution), not the first one (which might
        only contain tool_calls). When auto_execute_tools=False, returns the first
        Prediction with tool_calls for manual handling.

        Args:
            auto_execute_tools: If True, automatically execute tools and return
                final answer. If False, return Prediction with tool_calls for
                manual execution. Default: True.
            history: Optional History object for multi-turn conversations.
            **inputs: Input values for the module

        Returns:
            Final Prediction object (after all tool executions if auto_execute_tools=True)
        """
        return await self.aexecute(
            stream=False, auto_execute_tools=auto_execute_tools, history=history, **inputs
        )

    def _process_tool_call_delta(
        self, tool_calls: dict[int, dict[str, Any]], delta_tool_calls: list[Any]
    ) -> None:
        """Process tool call deltas and accumulate them.

        Args:
            tool_calls: Dictionary to accumulate tool calls in
            delta_tool_calls: List of tool call deltas from the chunk
        """
        for tool_call in delta_tool_calls:
            idx = tool_call.index
            if idx not in tool_calls:
                tool_calls[idx] = {
                    "id": tool_call.id or "",
                    "type": tool_call.type or "function",
                    "function": {
                        "name": tool_call.function.name if tool_call.function else "",
                        "arguments": "",
                    },
                }

            if tool_call.function and tool_call.function.arguments:
                tool_calls[idx]["function"]["arguments"] += tool_call.function.arguments

    async def _process_content_delta(
        self,
        delta: str,
        acc_delta: str,
        current_field: str | None,
        accumulated_content: dict[str, list[str]],
        output_fields: dict[str, Any],
        field_pattern: re.Pattern[str],
        queue: asyncio.Queue[StreamEvent | None],
    ) -> tuple[str, str | None]:
        """Process content delta and stream field chunks.

        Args:
            delta: New content delta
            acc_delta: Accumulated delta so far
            current_field: Current field being processed
            accumulated_content: Dictionary of accumulated content per field
            output_fields: Output fields from signature
            field_pattern: Regex pattern for field markers
            queue: Event queue to put chunks in

        Returns:
            Tuple of (updated acc_delta, updated current_field)
        """
        acc_delta += delta

        if not acc_delta:
            return acc_delta, current_field

        match = field_pattern.search(acc_delta)
        if match:
            if current_field:
                field_content = "".join(accumulated_content[current_field])
                await queue.put(
                    StreamChunk(self, current_field, "", field_content, is_complete=True)
                )

            current_field = match.group(1)
            acc_delta = field_pattern.sub("", acc_delta)
            if acc_delta.startswith("\n"):
                acc_delta = acc_delta[1:]

        if (
            current_field
            and current_field in output_fields
            and not field_pattern.match(acc_delta, partial=True)
        ):
            accumulated_content[current_field].append(acc_delta)
            field_content = "".join(accumulated_content[current_field])
            await queue.put(
                StreamChunk(self, current_field, acc_delta, field_content, is_complete=False)
            )
            acc_delta = ""

        return acc_delta, current_field

    def _execute_lm_callbacks(
        self,
        stage: str,
        call_id: str,
        inputs: dict | None = None,
        outputs: dict | None = None,
        exception: Exception | None = None,
    ) -> None:
        """Execute LM callbacks for start/end events.

        Args:
            stage: "start" or "end"
            call_id: Unique call identifier
            inputs: Input parameters for LM call (for start)
            outputs: Output from LM call (for end)
            exception: Exception if LM call failed (for end)
        """
        from udspy.callback import BaseCallback

        # Get combined global and instance-level callbacks
        global_callbacks = settings.get("callbacks", [])
        instance_callbacks = getattr(self, "callbacks", [])
        callbacks = global_callbacks + instance_callbacks

        for callback in callbacks:
            if not isinstance(callback, BaseCallback):
                continue

            try:
                if stage == "start" and inputs is not None:
                    callback.on_lm_start(call_id=call_id, instance=self, inputs=inputs)
                elif stage == "end":
                    callback.on_lm_end(call_id=call_id, outputs=outputs, exception=exception)
            except Exception as e:
                logger.warning(
                    f"Error in callback {callback.__class__.__name__}.on_lm_{stage}: {e}"
                )

    async def _process_nonstreaming(
        self, completion_kwargs: dict[str, Any], should_emit: bool
    ) -> Prediction:
        """Process non-streaming LLM call.

        Args:
            completion_kwargs: Arguments for the completion API call
            should_emit: If True, emit events to active queue

        Returns:
            Prediction object
        """
        import uuid

        from udspy.streaming import _stream_queue

        # Start LM callbacks
        call_id = uuid.uuid4().hex
        self._execute_lm_callbacks("start", call_id, inputs=completion_kwargs)

        client = settings.aclient
        outputs_dict = None
        exception = None

        try:
            response = await client.chat.completions.create(**completion_kwargs)
            outputs_dict = {
                "response": (
                    response.model_dump() if hasattr(response, "model_dump") else str(response)
                )
            }

            message = response.choices[0].message
            completion_text = message.content or ""
            tool_calls_data = message.tool_calls

            outputs = self.adapter.parse_outputs(self.signature, completion_text)
        except Exception as e:
            exception = e
            raise
        finally:
            self._execute_lm_callbacks("end", call_id, outputs=outputs_dict, exception=exception)

        if tool_calls_data:
            outputs["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in tool_calls_data
            ]

        prediction = Prediction(**outputs)

        if should_emit:
            queue = _stream_queue.get()
            if queue is not None:
                output_fields = self.signature.get_output_fields()
                for field_name in output_fields:
                    if hasattr(prediction, field_name):
                        field_value = getattr(prediction, field_name)
                        if field_value:
                            from udspy.streaming import StreamChunk

                            await queue.put(
                                StreamChunk(
                                    self, field_name, field_value, field_value, is_complete=True
                                )
                            )

                await queue.put(prediction)

        return prediction

    async def _process_streaming(
        self, completion_kwargs: dict[str, Any], should_emit: bool
    ) -> Prediction:
        """Process streaming LLM call.

        Args:
            completion_kwargs: Arguments for the completion API call
            should_emit: If True, emit events to active queue from context

        Returns:
            Prediction object
        """
        from udspy.streaming import _stream_queue

        active_queue = _stream_queue.get()

        if should_emit and active_queue is not None:
            return await self._process_llm_stream(
                active_queue, completion_kwargs, emit_sentinel=False, emit_prediction=False
            )
        else:
            queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
            llm_task = asyncio.create_task(
                self._process_llm_stream(queue, completion_kwargs, emit_sentinel=True)
            )

            final_prediction: Prediction | None = None
            while True:
                event = await queue.get()
                if event is None:
                    break

                if isinstance(event, Prediction):
                    final_prediction = event

            await llm_task

            if final_prediction is None:
                raise RuntimeError("No prediction generated from stream")

            return final_prediction

    async def _process_llm_stream(
        self,
        queue: asyncio.Queue[StreamEvent | None],
        completion_kwargs: dict[str, Any],
        emit_sentinel: bool = True,
        emit_prediction: bool = True,
    ) -> Prediction:
        """Background task to process LLM stream and put events in queue.

        Args:
            queue: Event queue to put events in
            completion_kwargs: Arguments for the completion API call
            emit_sentinel: If True, emit None sentinel at the end
            emit_prediction: If True, emit final Prediction to queue

        Returns:
            Final Prediction object
        """
        import uuid

        # Start LM callbacks
        call_id = uuid.uuid4().hex
        self._execute_lm_callbacks("start", call_id, inputs=completion_kwargs)

        outputs_dict = None
        exception = None

        try:
            client = settings.aclient
            stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
                **completion_kwargs
            )

            output_fields = self.signature.get_output_fields()
            field_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")
            current_field: str | None = None
            accumulated_content: dict[str, list[str]] = {name: [] for name in output_fields}
            full_completion: list[str] = []
            acc_delta: str = ""
            tool_calls: dict[int, dict[str, Any]] = {}

            async for chunk in stream:
                choice = chunk.choices[0]

                if choice.delta.tool_calls:
                    self._process_tool_call_delta(tool_calls, choice.delta.tool_calls)

                delta = choice.delta.content or ""
                if delta:
                    full_completion.append(delta)
                    acc_delta, current_field = await self._process_content_delta(
                        delta,
                        acc_delta,
                        current_field,
                        accumulated_content,
                        output_fields,
                        field_pattern,
                        queue,
                    )

            if current_field:
                field_content = "".join(accumulated_content[current_field])
                await queue.put(
                    StreamChunk(self, current_field, "", field_content, is_complete=True)
                )

            completion_text = "".join(full_completion)
            outputs = self.adapter.parse_outputs(self.signature, completion_text)

            if tool_calls:
                outputs["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                    for tc in tool_calls.values()
                ]

            prediction = Prediction(**outputs)
            if emit_prediction:
                await queue.put(prediction)

            outputs_dict = {
                "prediction": prediction.model_dump()
                if hasattr(prediction, "model_dump")
                else str(prediction)
            }
            return prediction

        except Exception as e:
            import traceback

            exception = e
            error_event = type(
                "StreamError",
                (StreamEvent,),
                {"error": str(e), "traceback": traceback.format_exc()},
            )()
            await queue.put(error_event)
            raise
        finally:
            # End LM callbacks
            self._execute_lm_callbacks("end", call_id, outputs=outputs_dict, exception=exception)
            if emit_sentinel:
                await queue.put(None)
