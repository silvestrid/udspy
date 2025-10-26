"""Tests for settings and context management."""

from openai import AsyncOpenAI

from udspy import settings


def test_configure_with_api_key() -> None:
    """Test configuring settings with API key."""
    settings.configure(api_key="sk-test-key", model="gpt-4")

    assert settings.default_model == "gpt-4"
    assert isinstance(settings.aclient, AsyncOpenAI)


def test_configure_with_custom_client() -> None:
    """Test configuring with custom async client."""
    custom_aclient = AsyncOpenAI(api_key="sk-custom")

    settings.configure(aclient=custom_aclient)

    assert settings.aclient == custom_aclient


def test_context_override_model() -> None:
    """Test context manager overrides model."""
    settings.configure(api_key="sk-global", model="gpt-4o-mini")

    assert settings.default_model == "gpt-4o-mini"

    with settings.context(model="gpt-4"):
        assert settings.default_model == "gpt-4"

    # Back to global settings
    assert settings.default_model == "gpt-4o-mini"


def test_context_override_api_key() -> None:
    """Test context manager creates new client with different API key."""
    settings.configure(api_key="sk-global")
    global_aclient = settings.aclient

    with settings.context(api_key="sk-context"):
        context_aclient = settings.aclient
        assert context_aclient != global_aclient

    # Back to global client
    assert settings.aclient == global_aclient


def test_context_override_kwargs() -> None:
    """Test context manager overrides default kwargs."""
    settings.configure(api_key="sk-test", temperature=0.5)

    assert settings.default_kwargs["temperature"] == 0.5

    with settings.context(temperature=0.9, max_tokens=100):
        kwargs = settings.default_kwargs
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 100

    # Back to global kwargs
    assert settings.default_kwargs["temperature"] == 0.5
    assert "max_tokens" not in settings.default_kwargs


def test_nested_contexts() -> None:
    """Test nested context managers."""
    settings.configure(api_key="sk-global", model="gpt-4o-mini")

    assert settings.default_model == "gpt-4o-mini"

    with settings.context(model="gpt-4"):
        assert settings.default_model == "gpt-4"

        with settings.context(model="gpt-4-turbo"):
            assert settings.default_model == "gpt-4-turbo"

        # Back to outer context
        assert settings.default_model == "gpt-4"

    # Back to global
    assert settings.default_model == "gpt-4o-mini"


def test_context_with_custom_client() -> None:
    """Test context manager with custom async client."""
    settings.configure(api_key="sk-global")

    custom_aclient = AsyncOpenAI(api_key="sk-custom")

    with settings.context(aclient=custom_aclient):
        assert settings.aclient == custom_aclient

    # Back to global clients
    assert settings.aclient != custom_aclient


def test_context_preserves_client_when_only_changing_other_settings() -> None:
    """Test that client is preserved when context only changes model/callbacks/kwargs."""
    settings.configure(api_key="sk-global", model="gpt-4o-mini")
    original_client = settings.aclient

    # Test 1: Only changing model should keep the same client
    with settings.context(model="gpt-4"):
        assert settings.aclient is original_client
        assert settings.default_model == "gpt-4"

    # Test 2: Only changing callbacks should keep the same client
    from udspy import BaseCallback

    class TestCallback(BaseCallback):
        pass

    with settings.context(callbacks=[TestCallback()]):
        assert settings.aclient is original_client

    # Test 3: Only changing kwargs should keep the same client
    with settings.context(temperature=0.9, max_tokens=100):
        assert settings.aclient is original_client
        assert settings.default_kwargs["temperature"] == 0.9

    # Test 4: Changing multiple non-client settings should keep the same client
    with settings.context(model="gpt-4-turbo", temperature=0.5, callbacks=[TestCallback()]):
        assert settings.aclient is original_client
        assert settings.default_model == "gpt-4-turbo"

    # Test 5: Providing api_key SHOULD create a new client
    with settings.context(api_key="sk-context"):
        assert settings.aclient is not original_client

    # Test 6: Providing base_url SHOULD create a new client
    with settings.context(base_url="http://localhost:8000"):
        assert settings.aclient is not original_client

    # Test 7: Providing aclient SHOULD use the provided client
    custom_client = AsyncOpenAI(api_key="sk-custom")
    with settings.context(aclient=custom_client):
        assert settings.aclient is custom_client
        assert settings.aclient is not original_client

    # After all contexts, should be back to original client
    assert settings.aclient is original_client
