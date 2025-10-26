"""Base classes for modules."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from udspy.callback import with_callbacks
from udspy.confirmation import ConfirmationRequired
from udspy.streaming import Prediction, StreamEvent
from udspy.utils import ensure_sync_context


class Module:
    """Base class for all udspy modules.

    Modules are composable async-first units. The core method is `aexecute()`
    which handles both streaming and non-streaming execution. Public methods
    `astream()` and `aforward()` are thin wrappers around `aexecute()`.

    Subclasses should implement `aexecute()` to define their behavior.

    Example:
        ```python
        # Async streaming (real-time)
        async for event in module.astream(question="What is AI?"):
            if isinstance(event, StreamChunk):
                print(event.delta, end="", flush=True)
            elif isinstance(event, Prediction):
                result = event

        # Async non-streaming
        result = await module.aforward(question="What is AI?")

        # Sync (for scripts, notebooks)
        result = module(question="What is AI?")
        result = module.forward(question="What is AI?")
        ```
    """

    def __init__(self, *, callbacks: list[Any] | None = None):
        """Initialize module.

        Args:
            callbacks: Optional list of callback handlers for this module instance
        """
        self.callbacks = callbacks or []

    @with_callbacks
    async def aexecute(self, *, stream: bool = False, **inputs: Any) -> Prediction:
        """Core execution method. Must be implemented by subclasses.

        This is the single implementation point for both streaming and non-streaming
        execution. It always returns a Prediction, and optionally emits StreamEvent
        objects to the active queue (if one exists in the context).

        Args:
            stream: If True, request streaming from LLM provider. If False, use
                non-streaming API calls.
            **inputs: Input values for the module

        Returns:
            Final Prediction object

        Behavior:
            - Checks for active stream queue via _stream_queue.get()
            - If queue exists: emits StreamChunk and Prediction events
            - Always returns final Prediction (even in streaming mode)
            - This enables composability: nested modules emit events automatically

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement aexecute() method")

    async def astream(self, **inputs: Any) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming method. Sets up queue and yields events.

        This method sets up the stream queue context, calls aexecute() with
        streaming enabled, and yields all events from the queue.

        Args:
            **inputs: Input values for the module

        Yields:
            StreamEvent objects (StreamChunk, Prediction, and custom events)
        """
        from udspy.streaming import _stream_queue

        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        token = _stream_queue.set(queue)

        try:
            task = asyncio.create_task(self.aexecute(stream=True, **inputs))

            while True:
                if task.done():
                    try:
                        await task
                    except Exception:
                        raise

                    while not queue.empty():
                        event = queue.get_nowait()
                        if event is not None:
                            yield event
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                except TimeoutError:
                    continue

                if event is None:
                    break
                yield event

        finally:
            try:
                _stream_queue.reset(token)
            except (ValueError, LookupError):
                pass

    async def aforward(self, *, resume_state: Any = None, **inputs: Any) -> Prediction:
        """Async non-streaming method. Returns final result directly.

        This method calls aexecute() with streaming disabled. If called from
        within a streaming context (i.e., another module is streaming), events
        will still be emitted to the active queue.

        Supports resuming from a ConfirmationRequired exception by providing
        resume_state. This enables loop-based confirmation handling.

        Args:
            resume_state: Optional ResumeState containing exception and user response.
                Can also be a raw ConfirmationRequired exception (will use "yes" as response).
            **inputs: Input values for the module

        Returns:
            Final Prediction object

        Example:
            ```python
            from udspy import ResumeState

            # Loop-based confirmation handling
            resume_state = None

            while True:
                try:
                    result = await agent.aforward(
                        question="Delete files",
                        resume_state=resume_state
                    )
                    break
                except ConfirmationRequired as e:
                    user_response = input(f"{e.question} (yes/no): ")
                    resume_state = ResumeState(e, user_response)
            ```
        """
        if resume_state is not None:
            # Support both ResumeState and raw ConfirmationRequired for backward compat
            from udspy.confirmation import ConfirmationRequired, ResumeState

            if isinstance(resume_state, ResumeState):
                return await self.aresume(resume_state.user_response, resume_state.exception)
            elif isinstance(resume_state, ConfirmationRequired):
                # Backward compatibility: treat raw exception as "yes" response
                return await self.aresume("yes", resume_state)
            else:
                # Assume it's the old-style state, try to resume
                return await self.aresume("yes", resume_state)
        return await self.aexecute(stream=False, **inputs)

    def forward(self, *, resume_state: Any = None, **inputs: Any) -> Prediction:
        """Sync non-streaming method. Wraps aforward() with async_to_sync.

        This provides sync compatibility for scripts and notebooks. Cannot be
        called from within an async context (use aforward() instead).

        Supports resuming from a ConfirmationRequired exception by providing
        resume_state. This enables loop-based confirmation handling.

        Args:
            resume_state: Optional ResumeState containing exception and user response.
                Can also be a raw ConfirmationRequired exception (will use "yes" as response).
            **inputs: Input values for the module (includes both input fields
                and any module-specific parameters like auto_execute_tools)

        Returns:
            Final Prediction object

        Raises:
            RuntimeError: If called from within an async context

        Example:
            ```python
            from udspy import ResumeState

            # Loop-based confirmation handling
            resume_state = None

            while True:
                try:
                    result = agent.forward(
                        question="Delete files",
                        resume_state=resume_state
                    )
                    break
                except ConfirmationRequired as e:
                    user_response = input(f"{e.question} (yes/no): ")
                    resume_state = ResumeState(e, user_response)
            ```
        """
        ensure_sync_context(f"{self.__class__.__name__}.forward")
        return asyncio.run(self.aforward(resume_state=resume_state, **inputs))

    def __call__(self, *, resume_state: Any = None, **inputs: Any) -> Prediction:
        """Sync convenience method. Calls forward().

        Supports resuming from a ConfirmationRequired exception by providing
        resume_state. This enables loop-based confirmation handling.

        Args:
            resume_state: Optional ResumeState containing exception and user response.
                Can also be a raw ConfirmationRequired exception (will use "yes" as response).
            **inputs: Input values for the module

        Returns:
            Final Prediction object

        Example:
            ```python
            from udspy import ResumeState

            # Loop-based confirmation handling
            resume_state = None

            while True:
                try:
                    result = agent(
                        question="Delete files",
                        resume_state=resume_state
                    )
                    break
                except ConfirmationRequired as e:
                    user_response = input(f"{e.question} (yes/no): ")
                    resume_state = ResumeState(e, user_response)
            ```
        """
        return self.forward(resume_state=resume_state, **inputs)

    async def asuspend(self, exception: ConfirmationRequired) -> Any:
        """Async suspend execution and save state.

        Called when ConfirmationRequired is raised. Subclasses should override
        to save any module-specific state needed for resumption.

        Args:
            exception: The ConfirmationRequired exception that was raised

        Returns:
            Saved state (can be any type, will be passed to aresume)
        """
        # Default implementation returns the exception itself as state
        return exception

    def suspend(self, exception: ConfirmationRequired) -> Any:
        """Sync suspend execution and save state.

        Wraps asuspend() with async_to_sync.

        Args:
            exception: The ConfirmationRequired exception that was raised

        Returns:
            Saved state (can be any type, will be passed to resume)
        """
        ensure_sync_context(f"{self.__class__.__name__}.suspend")
        return asyncio.run(self.asuspend(exception))

    async def aresume(self, user_response: str, saved_state: Any) -> Prediction:
        """Async resume execution after user input.

        Called to resume execution after a ConfirmationRequired exception.
        Subclasses must override to implement resumption logic.

        Args:
            user_response: The user's response. Can be:
                - "yes"/"y" to approve the action
                - "no"/"n" to reject the action
                - "feedback" to provide feedback for LLM re-reasoning
                - JSON string with "edit" to modify tool arguments
            saved_state: State returned from asuspend()

        Returns:
            Final Prediction object

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement aresume() method")

    def resume(self, user_response: str, saved_state: Any) -> Prediction:
        """Sync resume execution after user input.

        Wraps aresume() with async_to_sync.

        Args:
            user_response: The user's response
            saved_state: State returned from suspend()

        Returns:
            Final Prediction object
        """
        ensure_sync_context(f"{self.__class__.__name__}.resume")
        return asyncio.run(self.aresume(user_response, saved_state))
