"""ReAct module for reasoning and acting with tools."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from udspy.callback import with_callbacks
from udspy.confirmation import ConfirmationRequired, respond_to_confirmation
from udspy.module.base import Module
from udspy.module.chain_of_thought import ChainOfThought
from udspy.module.predict import Predict
from udspy.signature import Signature, make_signature
from udspy.streaming import Prediction

if TYPE_CHECKING:
    from udspy.tool import Tool

logger = logging.getLogger(__name__)


class ReAct(Module):
    """ReAct (Reasoning and Acting) module for tool-using agents.

    ReAct iteratively reasons about the current situation and decides whether
    to call a tool or finish the task. Key features:

    - Iterative reasoning with tool execution
    - Built-in ask_to_user tool for clarification
    - Tool confirmation support for confirmations
    - State saving/restoration for resumption after confirmation requests
    - Real-time streaming of reasoning and tool usage

    Example (Basic Usage):
        ```python
        from udspy import ReAct, Signature, InputField, OutputField, tool
        from pydantic import Field

        @tool(name="search", description="Search for information")
        def search(query: str = Field(...)) -> str:
            return f"Results for: {query}"

        class QA(Signature):
            '''Answer questions using available tools.'''
            question: str = InputField()
            answer: str = OutputField()

        react = ReAct(QA, tools=[search])
        result = react(question="What is the weather in Tokyo?")
        ```

    Example (Streaming):
        ```python
        # Stream the agent's reasoning process in real-time
        async for event in react.astream(question="What is Python?"):
            if isinstance(event, StreamChunk):
                print(event.delta, end="", flush=True)
            elif isinstance(event, Prediction):
                print(f"Answer: {event.answer}")
        ```

        See examples/react_streaming.py for a complete streaming example.

    Example (Tools with Confirmation):
        ```python
        from udspy import ConfirmationRequired, ConfirmationRejected

        @tool(name="delete_file", require_confirmation=True)
        def delete_file(path: str = Field(...)) -> str:
            return f"Deleted {path}"

        react = ReAct(QA, tools=[delete_file])

        try:
            result = await react.aforward(question="Delete /tmp/test.txt")
        except ConfirmationRequired as e:
            # User is asked for confirmation
            print(f"Confirm: {e.question}")
            # Approve: respond_to_confirmation(e.confirmation_id, approved=True)
            # Reject: respond_to_confirmation(e.confirmation_id, approved=False, status="rejected")
            result = await react.aresume("yes", e)
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        tools: list[Callable | Tool],
        *,
        max_iters: int = 10,
        enable_ask_to_user: bool = True,
        callbacks: list[Any] | None = None,
    ):
        """Initialize ReAct module.

        Args:
            signature: Signature defining inputs and outputs, or signature string
            tools: List of tool functions (decorated with @tool) or Tool objects
            max_iters: Maximum number of reasoning iterations (default: 10)
            enable_ask_to_user: Whether to enable ask_to_user tool (default: True)
            callbacks: Optional list of callback handlers for this module instance
        """
        from udspy.tool import Tool

        super().__init__(callbacks=callbacks)

        # Convert string signature to Signature class
        if isinstance(signature, str):
            signature = Signature.from_string(signature)

        self.signature = signature

        self.max_iters = max_iters
        self.enable_ask_to_user = enable_ask_to_user

        tool_list = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tool_list}

        inputs = ", ".join([f"`{k}`" for k in self.signature.get_input_fields().keys()])
        outputs = ", ".join([f"`{k}`" for k in self.signature.get_output_fields().keys()])

        base_instructions = getattr(self.signature, "__doc__", "")
        instr = [f"{base_instructions}\n"] if base_instructions else []

        instr.extend(
            [
                f"You are an Agent. You will be given {inputs} as input.",
                f"Your goal is to use the supplied tools to accomplish the task and produce {outputs}.\n",
                "Think step-by-step about what to do next, then call the appropriate tool.",
                "Always explain your reasoning before calling a tool.",
                "When you have enough information, call the 'finish' tool to complete the task.",
            ]
        )

        def finish_tool(**_kwargs: Any) -> str:  # pyright: ignore[reportUnusedParameter]
            """Finish tool that accepts and ignores any arguments."""
            return "Task completed"

        self.tools["finish"] = Tool(
            func=finish_tool,
            name="finish",
            desc=f"Call this when you have all information needed to produce {outputs}",
            args={},
        )

        if self.enable_ask_to_user:

            def ask_to_user_impl(question: str) -> str:  # noqa: ARG001
                """Ask the user for clarification."""
                return ""

            self.tools["ask_to_user"] = Tool(
                func=ask_to_user_impl,
                name="ask_to_user",
                description="Ask the user for clarification when needed. Use this when you need more information or when the request is ambiguous.",
                require_confirmation=True,
            )

        react_input_fields: dict[str, type] = {}
        for name, field_info in self.signature.get_input_fields().items():
            react_input_fields[name] = field_info.annotation or str

        react_input_fields["trajectory"] = str

        react_output_fields: dict[str, type] = {
            "reasoning": str,
        }

        self.react_signature = make_signature(
            react_input_fields,
            react_output_fields,
            "\n".join(instr),
        )

        extract_input_fields: dict[str, type] = {}
        extract_output_fields: dict[str, type] = {}

        for name, field_info in self.signature.get_input_fields().items():
            extract_input_fields[name] = field_info.annotation or str

        for name, field_info in self.signature.get_output_fields().items():
            extract_output_fields[name] = field_info.annotation or str

        extract_input_fields["trajectory"] = str

        self.extract_signature = make_signature(
            extract_input_fields,
            extract_output_fields,
            base_instructions or "Extract the final answer from the trajectory",
        )

        self.react_module = Predict(
            self.react_signature, tools=list(self.tools.values()), callbacks=callbacks
        )
        self.extract_module = ChainOfThought(self.extract_signature, callbacks=callbacks)

    def _format_trajectory(self, trajectory: dict[str, Any]) -> str:
        """Format trajectory as a string for the LLM.

        Args:
            trajectory: Dictionary with reasoning_N, tool_name_N, tool_args_N, observation_N keys

        Returns:
            Formatted string representation
        """
        if not trajectory:
            return "No actions taken yet."

        lines = []
        iteration = 0
        while f"observation_{iteration}" in trajectory:
            lines.append(f"\n--- Step {iteration + 1} ---")
            if f"reasoning_{iteration}" in trajectory:
                lines.append(f"Reasoning: {trajectory[f'reasoning_{iteration}']}")
            if f"tool_name_{iteration}" in trajectory:
                lines.append(f"Tool: {trajectory[f'tool_name_{iteration}']}")
            if f"tool_args_{iteration}" in trajectory:
                lines.append(f"Args: {json.dumps(trajectory[f'tool_args_{iteration}'])}")
            lines.append(f"Observation: {trajectory[f'observation_{iteration}']}")
            iteration += 1

        return "\n".join(lines)

    async def _execute_iteration(
        self,
        idx: int,
        trajectory: dict[str, Any],
        input_args: dict[str, Any],
        *,
        stream: bool = False,
        pending_tool_call: dict[str, Any] | None = None,
    ) -> bool:
        """Execute a single ReAct iteration.

        Args:
            idx: Current iteration index
            trajectory: Current trajectory state
            input_args: Original input arguments
            pending_tool_call: Optional pending tool call to execute (for resumption)
                             Format: {"name": str, "args": dict, "id": str}

        Returns:
            should_stop: Whether to stop the ReAct loop

        Raises:
            ConfirmationRequired: When human input is needed
        """
        if pending_tool_call:
            tool_name = pending_tool_call["name"]
            tool_args = pending_tool_call["args"]
            tool_call_id = pending_tool_call.get("id", "")

            trajectory[f"tool_name_{idx}"] = tool_name
            trajectory[f"tool_args_{idx}"] = tool_args

            try:
                tool = self.tools[tool_name]
                if inspect.iscoroutinefunction(tool.func):
                    observation = await tool.func(**tool_args)
                elif tool.require_confirmation:
                    observation = tool.func(**tool_args)
                else:
                    import asyncio

                    loop = asyncio.get_event_loop()
                    observation = await loop.run_in_executor(None, lambda: tool.func(**tool_args))
            except ConfirmationRequired as e:
                e.context = {
                    "trajectory": trajectory.copy(),
                    "iteration": idx,
                    "input_args": input_args.copy(),
                }
                if e.tool_call and tool_call_id:
                    e.tool_call.call_id = tool_call_id
                raise
            except Exception as e:
                observation = f"Error executing {tool_name}: {str(e)}"
                logger.warning(f"Tool execution failed: {e}")

            trajectory[f"observation_{idx}"] = str(observation)
            should_stop = tool_name == "finish"
            return should_stop

        formatted_trajectory = self._format_trajectory(trajectory)
        pred = await self.react_module.aexecute(
            stream=stream,
            **input_args,
            trajectory=formatted_trajectory,
            auto_execute_tools=False,
        )

        reasoning = pred.get("reasoning", "")
        trajectory[f"reasoning_{idx}"] = reasoning

        if "tool_calls" not in pred or not pred.tool_calls:
            logger.debug(
                f"No tool calls in prediction. Keys: {list(pred.keys())}, pred: {dict(pred)}"
            )
            raise ValueError("LLM did not call any tools")

        tool_call = pred.tool_calls[0]
        tool_name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id", "")

        tool_args_str = tool_call.get("arguments", "{}")
        try:
            tool_args = json.loads(tool_args_str) if tool_args_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool arguments: {tool_args_str}")
            tool_args = {}

        logger.debug(f"Tool call - name: {tool_name}, args: {tool_args}, id: {tool_call_id}")

        trajectory[f"tool_name_{idx}"] = tool_name
        trajectory[f"tool_args_{idx}"] = tool_args

        try:
            tool = self.tools[tool_name]
            observation = await tool.acall(**tool_args)
        except ConfirmationRequired as e:
            e.context = {
                "trajectory": trajectory.copy(),
                "iteration": idx,
                "input_args": input_args.copy(),
            }
            if e.tool_call and tool_call_id:
                e.tool_call.call_id = tool_call_id
            raise
        except Exception as e:
            observation = f"Error executing {tool_name}: {str(e)}"
            logger.warning(f"Tool execution failed: {e}")

        trajectory[f"observation_{idx}"] = str(observation)

        should_stop = tool_name == "finish"
        return should_stop

    @with_callbacks
    async def aexecute(self, *, stream: bool = False, **input_args: Any) -> Prediction:
        """Execute ReAct loop.

        Args:
            stream: Passed to sub-modules
            **input_args: Input values matching signature's input fields

        Returns:
            Prediction with trajectory and output fields

        Raises:
            ConfirmationRequired: When human input is needed
        """
        max_iters = input_args.pop("max_iters", self.max_iters)
        trajectory: dict[str, Any] = {}

        for idx in range(max_iters):
            try:
                should_stop = await self._execute_iteration(
                    idx,
                    trajectory,
                    input_args,
                    stream=stream,
                )
                if should_stop:
                    break

            except ValueError as e:
                logger.warning(f"Agent failed to select valid tool: {e}")
                trajectory[f"observation_{idx}"] = f"Error: {e}"
                break

        formatted_trajectory = self._format_trajectory(trajectory)
        extract = await self.extract_module.aexecute(
            stream=stream,
            **input_args,
            trajectory=formatted_trajectory,
        )

        result_dict = {"trajectory": trajectory}
        for field_name in self.signature.get_output_fields():
            if hasattr(extract, field_name):
                result_dict[field_name] = getattr(extract, field_name)

        return Prediction(**result_dict)

    async def aforward(self, **input_args: Any) -> Prediction:
        return await self.aexecute(stream=False, **input_args)

    async def aresume(
        self,
        user_response: str,
        saved_state: Any,
    ) -> Prediction:
        """Async resume execution after user input.

        Args:
            user_response: The user's response. Can be:
                - "yes"/"y" to confirm tool execution with original args
                - "no"/"n" to reject and continue
                - JSON dict string to execute tool with modified args
                - Any other text is treated as user feedback for LLM to re-reason

        Returns:
            Final prediction after resuming

        Raises:
            ConfirmationRequired: If another human input is needed
        """
        trajectory = saved_state.context.get("trajectory", {}).copy()
        start_idx = saved_state.context.get("iteration", 0)
        input_args = saved_state.context.get("input_args", {}).copy()

        user_response_lower = user_response.lower().strip()
        pending_tool_call: dict[str, Any] | None = None

        if user_response_lower in ("yes", "y"):
            if saved_state.confirmation_id:
                respond_to_confirmation(
                    saved_state.confirmation_id, approved=True, status="approved"
                )
            if saved_state.tool_call:
                pending_tool_call = {
                    "name": saved_state.tool_call.name,
                    "args": saved_state.tool_call.args.copy(),
                    "id": saved_state.tool_call.call_id or "",
                }
            else:
                pending_tool_call = None
        elif user_response_lower in ("no", "n"):
            if saved_state.confirmation_id:
                respond_to_confirmation(
                    saved_state.confirmation_id, approved=False, status="rejected"
                )
            trajectory[f"observation_{start_idx}"] = "User rejected the operation"
            start_idx += 1
        else:
            try:
                modified_args = json.loads(user_response)
                if isinstance(modified_args, dict):
                    if saved_state.confirmation_id:
                        respond_to_confirmation(
                            saved_state.confirmation_id,
                            approved=True,
                            data=modified_args,
                            status="edited",
                        )
                    if saved_state.tool_call:
                        pending_tool_call = {
                            "name": saved_state.tool_call.name,
                            "args": modified_args,
                            "id": saved_state.tool_call.call_id or "",
                        }
                    else:
                        pending_tool_call = None
                else:
                    if saved_state.confirmation_id:
                        respond_to_confirmation(
                            saved_state.confirmation_id, approved=False, status="feedback"
                        )
                    trajectory[f"observation_{start_idx}"] = f"User feedback: {user_response}"
                    start_idx += 1
            except json.JSONDecodeError:
                if saved_state.confirmation_id:
                    respond_to_confirmation(
                        saved_state.confirmation_id, approved=False, status="feedback"
                    )
                trajectory[f"observation_{start_idx}"] = f"User feedback: {user_response}"
                start_idx += 1

        for idx in range(start_idx, self.max_iters):
            try:
                should_stop = await self._execute_iteration(
                    idx,
                    trajectory,
                    input_args,
                    pending_tool_call=pending_tool_call,
                )
                pending_tool_call = None

                if should_stop:
                    break

            except ValueError as e:
                logger.warning(f"Agent failed: {e}")
                trajectory[f"observation_{idx}"] = f"Error: {e}"
                break

        formatted_trajectory = self._format_trajectory(trajectory)
        extract = await self.extract_module.aforward(
            **input_args,
            trajectory=formatted_trajectory,
        )

        result_dict = {"trajectory": trajectory}
        for field_name in self.signature.get_output_fields():
            if hasattr(extract, field_name):
                result_dict[field_name] = getattr(extract, field_name)

        return Prediction(**result_dict)
