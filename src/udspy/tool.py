"""Tool decorator for native function calling."""

import inspect
from collections.abc import Callable
from typing import Any, get_args, get_origin

from pydantic.fields import FieldInfo

from udspy.callback import with_callbacks


class Tool:
    """Wrapper for a tool function with metadata.

    Tools are callable functions that can be passed to Predict. The function
    signature and annotations are automatically converted to an OpenAI tool schema.

    Example:
        ```python
        @tool(name="Calculator", description="Perform arithmetic operations")
        def calculator(
            operation: str = Field(description="add, subtract, multiply, divide"),
            a: float = Field(description="First number"),
            b: float = Field(description="Second number"),
        ) -> float:
            ops = {"add": a + b, "multiply": a * b, ...}
            return ops[operation]

        predictor = Predict(QA, tools=[calculator])
        ```
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        *,
        require_confirmation: bool = False,
        desc: str | None = None,  # Alias for description (DSPy compatibility)
        args: dict[str, str] | None = None,  # Optional manual arg spec (DSPy compatibility)
        callbacks: list[Any] | None = None,
    ):
        """Initialize a Tool.

        Args:
            func: The function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            require_confirmation: If True, wraps function with @confirm_first decorator
            desc: Alias for description (for DSPy compatibility)
            args: Optional manual argument specification (for DSPy compatibility)
            callbacks: Optional list of callback handlers for this tool instance
        """
        self.name = name or func.__name__
        self.func = func
        self.callbacks = callbacks or []

        # Wrap with @confirm_first decorator if requested
        if require_confirmation:
            from udspy.confirmation import confirm_first as confirm_first_decorator

            # We need the @confirm_first decorator to see the correct __name__
            # So we create a wrapper with the tool name, then apply the decorator
            tool_name = self.name
            original_func = func
            sig = inspect.signature(func)

            # Create a dynamic wrapper with the same signature but different name
            # Use a single function definition to satisfy mypy's type checker
            if inspect.iscoroutinefunction(func):

                async def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return await original_func(*args, **kwargs)

            else:

                def tool_wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
                    return original_func(*args, **kwargs)

            # Copy signature and set correct name
            tool_wrapper.__name__ = tool_name
            tool_wrapper.__signature__ = sig  # type: ignore[attr-defined]

            # Now apply @confirm_first - it will see the correct __name__
            self.func = confirm_first_decorator(tool_wrapper)
        self.description = description or desc or inspect.getdoc(func) or ""
        self.require_confirmation = require_confirmation

        # Aliases for DSPy compatibility
        self.desc = self.description
        self.args: dict[str, str] = args or {}  # Will be populated below if not provided

        # Extract parameter schema from function signature
        sig = inspect.signature(func)
        self.parameters: dict[str, dict[str, Any]] = {}

        for param_name, param in sig.parameters.items():
            # Skip *args, **kwargs
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            param_info: dict[str, Any] = {
                "type": param.annotation if param.annotation != inspect.Parameter.empty else str,
                "description": None,
                "required": param.default == inspect.Parameter.empty,
            }

            # Extract Field metadata if present
            if isinstance(param.default, FieldInfo):
                param_info["description"] = param.default.description
                param_info["required"] = param.default.is_required()

            self.parameters[param_name] = param_info

            # Populate args dict for DSPy compatibility (if not manually provided)
            if not args:
                type_str = (
                    param_info["type"].__name__
                    if hasattr(param_info["type"], "__name__")
                    else str(param_info["type"])
                )
                desc_str = param_info["description"] or "No description"
                self.args[param_name] = f"{type_str} - {desc_str}"

    @with_callbacks
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function."""
        return self.func(*args, **kwargs)

    @with_callbacks
    async def acall(self, **kwargs: Any) -> Any:
        """Async call the wrapped function.

        If the function is async, awaits it. Otherwise, runs it in executor.
        If require_confirmation is True, the @confirm_first decorator handles confirmation.

        Args:
            **kwargs: Arguments to pass to the function

        Returns:
            Function result

        Raises:
            ConfirmationRequired: If require_confirmation is True and not approved
        """
        # Execute the function (confirm_first decorator handles confirmation if enabled)
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        else:
            # Run sync function in executor to avoid blocking
            import asyncio

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.func(**kwargs))

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI tool schema.

        Returns:
            OpenAI-compatible tool schema dictionary
        """
        properties = {}
        required = []

        for name, info in self.parameters.items():
            prop: dict[str, Any] = {"type": self._python_type_to_json_type(info["type"])}

            if info["description"]:
                prop["description"] = info["description"]

            properties[name] = prop

            if info["required"]:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _python_type_to_json_type(python_type: Any) -> str:
        """Convert Python type annotation to JSON Schema type.

        Args:
            python_type: Python type annotation

        Returns:
            JSON Schema type string
        """
        # Handle Optional[T] -> T | None
        origin = get_origin(python_type)
        if origin is not None:
            args = get_args(python_type)
            # For Union types (including Optional), try to get the non-None type
            if len(args) > 0:
                for arg in args:
                    if arg is not type(None):  # noqa: E721
                        python_type = arg
                        break

        # Map Python types to JSON Schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        return type_map.get(python_type, "string")


def tool(
    name: str | None = None,
    description: str | None = None,
    *,
    require_confirmation: bool = False,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator to mark a function as a tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        require_confirmation: If True, wraps function with @confirm_first decorator

    Returns:
        Decorator function

    Example:
        ```python
        @tool(name="Calculator", description="Perform arithmetic operations")
        def calculator(
            operation: str = Field(description="add, subtract, multiply, divide"),
            a: float = Field(description="First number"),
            b: float = Field(description="Second number"),
        ) -> float:
            ops = {
                "add": a + b,
                "subtract": a - b,
                "multiply": a * b,
                "divide": a / b if b != 0 else float("inf"),
            }
            return ops[operation]

        # Tool that requires confirmation (e.g., destructive operations)
        @tool(name="DeleteFile", description="Delete a file", require_confirmation=True)
        def delete_file(path: str = Field(...)) -> str:
            os.remove(path)
            return f"Deleted {path}"
        ```
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(
            func, name=name, description=description, require_confirmation=require_confirmation
        )

    return decorator
