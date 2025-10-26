"""Callback system for telemetry and monitoring.

This module provides a DSPy-compatible callback system for tracking LLM calls,
module executions, and tool invocations. Compatible with Opik, MLflow, and other
observability tools that support DSPy callbacks.
"""

import functools
import inspect
import logging
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

ACTIVE_CALL_ID: ContextVar[str | None] = ContextVar("active_call_id", default=None)

logger = logging.getLogger(__name__)


class BaseCallback:
    """Base class for callback handlers.

    Subclass this and implement the desired handlers to track LLM calls, module
    executions, and tool invocations. Compatible with DSPy callback interface.

    Example:
        ```python
        from udspy import settings
        from udspy.callback import BaseCallback

        class LoggingCallback(BaseCallback):
            def on_lm_start(self, call_id, instance, inputs):
                print(f"LLM called with: {inputs}")

            def on_lm_end(self, call_id, outputs, exception):
                if exception:
                    print(f"LLM failed: {exception}")
                else:
                    print(f"LLM returned: {outputs}")

            def on_tool_start(self, call_id, instance, inputs):
                print(f"Tool {instance.name} called with: {inputs}")

            def on_tool_end(self, call_id, outputs, exception):
                print(f"Tool returned: {outputs}")

        # Set globally
        settings.configure(callbacks=[LoggingCallback()])

        # Or per-module
        predictor = Predict(QA, callbacks=[LoggingCallback()])
        ```
    """

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Called when a module's forward() method starts.

        Args:
            call_id: Unique identifier for this call
            instance: The Module instance being called
            inputs: Input arguments as key-value pairs
        """
        pass

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        """Called when a module's forward() method completes.

        Args:
            call_id: Unique identifier for this call
            outputs: The module's output, or None if exception occurred
            exception: Exception raised during execution, if any
        """
        pass

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Called when an LLM call starts.

        Args:
            call_id: Unique identifier for this call
            instance: The LLM client or adapter instance
            inputs: LLM input parameters (messages, model, etc.)
        """
        pass

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """Called when an LLM call completes.

        Args:
            call_id: Unique identifier for this call
            outputs: LLM response, or None if exception occurred
            exception: Exception raised during execution, if any
        """
        pass

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ) -> None:
        """Called when a tool is invoked.

        Args:
            call_id: Unique identifier for this call
            instance: The Tool instance being called
            inputs: Tool input arguments as key-value pairs
        """
        pass

    def on_tool_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        """Called when a tool invocation completes.

        Args:
            call_id: Unique identifier for this call
            outputs: Tool output, or None if exception occurred
            exception: Exception raised during execution, if any
        """
        pass


def with_callbacks(fn: Callable) -> Callable:
    """Decorator to add callback functionality to methods.

    Automatically calls appropriate callback handlers before and after
    method execution. Handles both sync and async methods.

    The decorator determines which callback handlers to call based on the
    instance type (Module, Tool, etc.) and method name.
    """

    def _execute_start_callbacks(
        instance: Any,
        fn: Callable,
        call_id: str,
        callbacks: list[BaseCallback],
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Execute all start callbacks."""
        # Get function arguments
        inputs = inspect.getcallargs(fn, instance, *args, **kwargs)
        # Remove self/instance from inputs
        inputs.pop("self", None)
        inputs.pop("instance", None)

        for callback in callbacks:
            try:
                handler = _get_on_start_handler(callback, instance, fn)
                handler(call_id=call_id, instance=instance, inputs=inputs)
            except Exception as e:
                logger.warning(f"Error in callback {callback.__class__.__name__}.on_*_start: {e}")

    def _execute_end_callbacks(
        instance: Any,
        fn: Callable,
        call_id: str,
        results: Any,
        exception: Exception | None,
        callbacks: list[BaseCallback],
    ) -> None:
        """Execute all end callbacks."""
        for callback in callbacks:
            try:
                handler = _get_on_end_handler(callback, instance, fn)
                handler(call_id=call_id, outputs=results, exception=exception)
            except Exception as e:
                logger.warning(f"Error in callback {callback.__class__.__name__}.on_*_end: {e}")

    def _get_active_callbacks(instance: Any) -> list[BaseCallback]:
        """Get combined global and instance-level callbacks."""
        from udspy.settings import settings

        global_callbacks = settings.get("callbacks", [])
        instance_callbacks = getattr(instance, "callbacks", [])
        return global_callbacks + instance_callbacks

    # Handle async functions
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                return await fn(instance, *args, **kwargs)

            call_id = uuid.uuid4().hex

            _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

            # Set active call ID for nested tracking
            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)

            results = None
            exception = None
            try:
                results = await fn(instance, *args, **kwargs)
                return results
            except Exception as e:
                exception = e
                raise
            finally:
                ACTIVE_CALL_ID.set(parent_call_id)
                _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

        return async_wrapper

    # Handle sync functions
    else:

        @functools.wraps(fn)
        def sync_wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                return fn(instance, *args, **kwargs)

            call_id = uuid.uuid4().hex

            _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

            # Set active call ID for nested tracking
            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)

            results = None
            exception = None
            try:
                results = fn(instance, *args, **kwargs)
                return results
            except Exception as e:
                exception = e
                raise
            finally:
                ACTIVE_CALL_ID.set(parent_call_id)
                _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

        return sync_wrapper


def _get_on_start_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """Get the appropriate on_start handler based on instance type."""
    # Import here to avoid circular imports
    from udspy.module.base import Module
    from udspy.tool import Tool

    if isinstance(instance, Tool):
        return callback.on_tool_start

    # For modules, check if it's an LLM-related method
    # In udspy, we don't have a separate LM class, so we'll use on_module_start
    # But we can add on_lm_start for Predict module when calling OpenAI
    if isinstance(instance, Module):
        return callback.on_module_start

    # Default to module
    return callback.on_module_start


def _get_on_end_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """Get the appropriate on_end handler based on instance type."""
    # Import here to avoid circular imports
    from udspy.module.base import Module
    from udspy.tool import Tool

    if isinstance(instance, Tool):
        return callback.on_tool_end

    if isinstance(instance, Module):
        return callback.on_module_end

    # Default to module
    return callback.on_module_end


__all__ = ["BaseCallback", "with_callbacks", "ACTIVE_CALL_ID"]
