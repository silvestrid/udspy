"""Global settings and configuration."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from openai import AsyncOpenAI


class Settings:
    """Global settings for udspy.

    Since udspy is async-first, we only need the async OpenAI client.
    Sync wrappers (forward(), __call__()) use asyncio.run() internally,
    which works fine with the async client.
    """

    def __init__(self) -> None:
        self._aclient: AsyncOpenAI | None = None
        self._default_model: str | None = None
        self._default_kwargs: dict[str, Any] = {}
        self._callbacks: list[Any] = []

        # Context-specific overrides (thread-safe)
        self._context_aclient: ContextVar[AsyncOpenAI | None] = ContextVar(
            "context_aclient", default=None
        )
        self._context_model: ContextVar[str | None] = ContextVar("context_model", default=None)
        self._context_kwargs: ContextVar[dict[str, Any] | None] = ContextVar(
            "context_kwargs", default=None
        )
        self._context_callbacks: ContextVar[list[Any] | None] = ContextVar(
            "context_callbacks", default=None
        )

    def configure(
        self,
        api_key: str = "",
        base_url: str | None = None,
        model: str | None = None,
        aclient: AsyncOpenAI | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure global OpenAI client and defaults.

        Args:
            api_key: OpenAI API key (creates default async client)
            base_url: Base URL for OpenAI API
            model: Default model to use for all predictions
            aclient: Custom async OpenAI client
            callbacks: List of callback handlers for telemetry/monitoring
            **kwargs: Default kwargs for all chat completions (temperature, etc.)

        Example:
            ```python
            import udspy
            from udspy import BaseCallback

            class LoggingCallback(BaseCallback):
                def on_lm_start(self, call_id, instance, inputs):
                    print(f"LLM called with: {inputs}")

            # With API key and callbacks
            udspy.settings.configure(
                api_key="sk-...",
                model="gpt-4o",
                callbacks=[LoggingCallback()]
            )

            # With custom client
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key="sk-...", timeout=30.0)
            udspy.settings.configure(aclient=client, model="gpt-4o")
            ```
        """
        if aclient:
            self._aclient = aclient
        else:
            self._aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

        if model:
            self._default_model = model

        if callbacks is not None:
            self._callbacks = callbacks

        self._default_kwargs.update(kwargs)

    @property
    def aclient(self) -> AsyncOpenAI:
        """Get the async OpenAI client (context-aware).

        This is used by all module operations, both async and sync.
        Sync wrappers use asyncio.run() internally.
        """
        # Check context first
        context_aclient = self._context_aclient.get()
        if context_aclient is not None:
            return context_aclient

        # Fall back to global client
        if self._aclient is None:
            raise RuntimeError(
                "OpenAI client not configured. Call udspy.settings.configure() first."
            )
        return self._aclient

    @property
    def default_model(self) -> str:
        """Get the default model name (context-aware)."""
        # Check context first
        context_model = self._context_model.get()
        if context_model is not None:
            return context_model

        # Fall back to global model
        if self._default_model is None:
            raise ValueError(
                "No model configured. Call settings.configure(model='...') or set in context."
            )
        return self._default_model

    @property
    def default_kwargs(self) -> dict[str, Any]:
        """Get the default kwargs for chat completions (context-aware)."""
        # Start with global defaults
        result = self._default_kwargs.copy()

        # Override with context-specific kwargs if present
        context_kwargs = self._context_kwargs.get()
        if context_kwargs is not None:
            result.update(context_kwargs)

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key (for callback compatibility).

        Args:
            key: Setting key to retrieve
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        if key == "callbacks":
            # Check context first
            context_callbacks = self._context_callbacks.get()
            if context_callbacks is not None:
                return context_callbacks
            return self._callbacks
        return default

    @contextmanager
    def context(
        self,
        api_key: str = "",
        base_url: str | None = None,
        model: str | None = None,
        aclient: AsyncOpenAI | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[None]:
        """Context manager for temporary settings overrides.

        This is thread-safe and allows you to use different API keys, models,
        or other settings within a specific context. Useful for multi-tenant
        applications.

        Args:
            api_key: Temporary OpenAI API key (creates temporary client)
            model: Temporary model to use
            aclient: Temporary async OpenAI client
            callbacks: Temporary callback handlers
            **kwargs: Temporary kwargs for chat completions

        Example:
            ```python
            import udspy
            from udspy import Predict, Signature, InputField, OutputField

            # Global settings
            udspy.settings.configure(api_key="global-key", model="gpt-4o-mini")

            class QA(Signature):
                question: str = InputField()
                answer: str = OutputField()

            predictor = Predict(QA)

            # Temporary override for a specific context (e.g., different tenant)
            with udspy.settings.context(api_key="tenant-key", model="gpt-4"):
                result = predictor(question="...")  # Uses "tenant-key" and "gpt-4"

            # Back to global settings
            result = predictor(question="...")  # Uses "global-key" and "gpt-4o-mini"
            ```
        """
        # Save current context values
        prev_aclient = self._context_aclient.get()
        prev_model = self._context_model.get()
        prev_kwargs = self._context_kwargs.get()
        prev_callbacks = self._context_callbacks.get()

        try:
            # Set context-specific values
            if aclient:
                self._context_aclient.set(aclient)
            elif api_key or base_url:
                self._context_aclient.set(AsyncOpenAI(api_key=api_key, base_url=base_url))

            if model:
                self._context_model.set(model)

            if callbacks is not None:
                self._context_callbacks.set(callbacks)

            if kwargs:
                # Merge with previous context kwargs if any
                merged_kwargs = (prev_kwargs or {}).copy()
                merged_kwargs.update(kwargs)
                self._context_kwargs.set(merged_kwargs)

            yield

        finally:
            # Restore previous context values
            self._context_aclient.set(prev_aclient)
            self._context_model.set(prev_model)
            self._context_kwargs.set(prev_kwargs)
            self._context_callbacks.set(prev_callbacks)


# Global settings instance
settings = Settings()
