"""Callback system examples.

This example demonstrates how to use callbacks for telemetry, monitoring,
and observability in udspy. Callbacks are compatible with DSPy, so tools
like Opik and MLflow work seamlessly.
"""

import asyncio
import json
import time
from typing import Any

from pydantic import Field

from udspy import BaseCallback, InputField, OutputField, Predict, Signature, settings, tool

settings.configure(
    model="gpt-oss:120b-cloud",
    base_url="http://localhost:11434/v1",
)


# =============================================================================
# Example 1: Basic Logging Callback
# =============================================================================


class LoggingCallback(BaseCallback):
    """Simple callback that logs all events to console."""

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        print(f"\n[{call_id[:8]}] 🚀 Module {type(instance).__name__} started")
        print(
            f"  Inputs: {json.dumps({k: v for k, v in inputs.items() if k != 'inputs'}, indent=2)}"
        )

    def on_module_end(
        self, call_id: str, outputs: Any | None, exception: Exception | None = None
    ) -> None:
        if exception:
            print(f"[{call_id[:8]}] ❌ Module failed: {exception}")
        else:
            print(f"[{call_id[:8]}] ✅ Module completed")

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        model = inputs.get("model", "unknown")
        print(f"[{call_id[:8]}] 🤖 LLM call started (model: {model})")

    def on_lm_end(
        self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None
    ) -> None:
        if exception:
            print(f"[{call_id[:8]}] ❌ LLM call failed: {exception}")
        else:
            print(f"[{call_id[:8]}] ✅ LLM call completed")

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        print(f"[{call_id[:8]}] 🔧 Tool {instance.name} called")

    def on_tool_end(
        self, call_id: str, outputs: Any | None, exception: Exception | None = None
    ) -> None:
        if exception:
            print(f"[{call_id[:8]}] ❌ Tool failed: {exception}")
        else:
            print(f"[{call_id[:8]}] ✅ Tool completed")


def example_basic_logging():
    """Example 1: Basic logging callback."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Logging Callback")
    print("=" * 80)

    # Define a simple QA signature
    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    # Configure global callback
    with settings.context(callbacks=[LoggingCallback()]):
        predictor = Predict(QA)
        result = predictor(question="What is the capital of France?")
        print(f"\n📝 Final Answer: {result.answer}")


# =============================================================================
# Example 2: Performance Monitoring
# =============================================================================


class PerformanceMonitor(BaseCallback):
    """Track execution time and performance metrics."""

    def __init__(self):
        self.start_times = {}
        self.metrics = {"module_calls": 0, "lm_calls": 0, "tool_calls": 0, "total_time": 0.0}

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self.start_times[f"module_{call_id}"] = time.time()
        self.metrics["module_calls"] += 1

    def on_module_end(
        self, call_id: str, outputs: Any | None, exception: Exception | None = None
    ) -> None:
        start_time = self.start_times.pop(f"module_{call_id}", time.time())
        duration = time.time() - start_time
        self.metrics["total_time"] += duration
        print(f"⏱️  Module execution took {duration:.2f}s")

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self.start_times[f"lm_{call_id}"] = time.time()
        self.metrics["lm_calls"] += 1

    def on_lm_end(
        self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None
    ) -> None:
        start_time = self.start_times.pop(f"lm_{call_id}", time.time())
        duration = time.time() - start_time
        print(f"⏱️  LLM call took {duration:.2f}s")

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self.start_times[f"tool_{call_id}"] = time.time()
        self.metrics["tool_calls"] += 1

    def on_tool_end(
        self, call_id: str, outputs: Any | None, exception: Exception | None = None
    ) -> None:
        start_time = self.start_times.pop(f"tool_{call_id}", time.time())
        duration = time.time() - start_time
        print(f"⏱️  Tool execution took {duration:.2f}s")

    def report(self):
        """Print performance summary."""
        print("\n📊 Performance Report:")
        print(f"  Module calls: {self.metrics['module_calls']}")
        print(f"  LLM calls: {self.metrics['lm_calls']}")
        print(f"  Tool calls: {self.metrics['tool_calls']}")
        print(f"  Total time: {self.metrics['total_time']:.2f}s")


def example_performance_monitoring():
    """Example 2: Performance monitoring."""
    print("\n" + "=" * 80)
    print("Example 2: Performance Monitoring")
    print("=" * 80)

    @tool(name="search", description="Search for information")
    def search(query: str = Field(description="Search query")) -> str:
        """Mock search tool."""
        time.sleep(0.5)  # Simulate API call
        return f"Results for: {query}"

    class Research(Signature):
        """Research a topic using tools."""

        topic: str = InputField()
        summary: str = OutputField()

    monitor = PerformanceMonitor()

    with settings.context(callbacks=[monitor]):
        predictor = Predict(Research, tools=[search])
        result = predictor(topic="Python programming")
        print(f"\n📝 Summary: {result.summary}")

    monitor.report()


# =============================================================================
# Example 3: Cost Tracking
# =============================================================================


class CostTracker(BaseCallback):
    """Track API costs (simplified example)."""

    # Simplified pricing (actual prices vary)
    PRICING = {
        "gpt-4o": {"input": 0.01, "output": 0.03},  # per 1k tokens
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self):
        self.total_cost = 0.0
        self.call_costs = []

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        # Store model for this call
        model = inputs.get("model", "gpt-4o-mini")
        self.call_costs.append({"call_id": call_id, "model": model, "cost": 0.0})

    def on_lm_end(
        self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None
    ) -> None:
        if exception or not outputs:
            return

        # Find the call record
        call_record = next((c for c in self.call_costs if c["call_id"] == call_id), None)
        if not call_record:
            return

        # In a real implementation, parse token counts from response
        # For this example, we'll use a simplified estimate
        model = call_record["model"]
        pricing = self.PRICING.get(model, {"input": 0.001, "output": 0.002})

        # Simplified cost calculation (in reality, parse usage from response)
        estimated_input_tokens = 100
        estimated_output_tokens = 50
        cost = (
            estimated_input_tokens / 1000 * pricing["input"]
            + estimated_output_tokens / 1000 * pricing["output"]
        )

        call_record["cost"] = cost
        self.total_cost += cost
        print(f"💰 Call cost: ${cost:.6f}")

    def report(self):
        """Print cost summary."""
        print(f"\n💰 Total cost: ${self.total_cost:.6f}")
        print(f"   Number of LLM calls: {len(self.call_costs)}")


def example_cost_tracking():
    """Example 3: Cost tracking."""
    print("\n" + "=" * 80)
    print("Example 3: Cost Tracking")
    print("=" * 80)

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    tracker = CostTracker()

    with settings.context(callbacks=[tracker]):
        predictor = Predict(QA)
        result = predictor(question="Explain quantum computing")
        print(f"\n📝 Answer: {result.answer}")

    tracker.report()


# =============================================================================
# Example 4: Combining Multiple Callbacks
# =============================================================================


def example_multiple_callbacks():
    """Example 4: Using multiple callbacks together."""
    print("\n" + "=" * 80)
    print("Example 4: Multiple Callbacks")
    print("=" * 80)

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    # Combine logging, performance monitoring, and cost tracking
    logger = LoggingCallback()
    monitor = PerformanceMonitor()
    tracker = CostTracker()

    with settings.context(callbacks=[logger, monitor, tracker]):
        predictor = Predict(QA)
        result = predictor(question="What is machine learning?")
        print(f"\n📝 Final Answer: {result.answer}")

    # Print reports from monitoring callbacks
    print("\n" + "=" * 80)
    monitor.report()
    tracker.report()


# =============================================================================
# Example 5: Per-Instance Callbacks
# =============================================================================


def example_per_instance_callbacks():
    """Example 5: Different callbacks for different modules."""
    print("\n" + "=" * 80)
    print("Example 5: Per-Instance Callbacks")
    print("=" * 80)

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    # Create two predictors with different callbacks
    production_monitor = PerformanceMonitor()
    debug_logger = LoggingCallback()

    # Production predictor: only performance monitoring
    prod_predictor = Predict(QA, callbacks=[production_monitor])

    # Debug predictor: verbose logging
    debug_predictor = Predict(QA, callbacks=[debug_logger])

    print("\n--- Production Predictor (Performance Monitoring Only) ---")
    result1 = prod_predictor(question="What is AI?")
    print(f"Answer: {result1.answer}")

    print("\n--- Debug Predictor (Verbose Logging) ---")
    result2 = debug_predictor(question="What is AI?")
    print(f"Answer: {result2.answer}")

    production_monitor.report()


# =============================================================================
# Example 6: Async Callbacks
# =============================================================================


async def example_async_callbacks():
    """Example 6: Callbacks work with async execution."""
    print("\n" + "=" * 80)
    print("Example 6: Async Callbacks")
    print("=" * 80)

    class QA(Signature):
        """Answer questions."""

        question: str = InputField()
        answer: str = OutputField()

    monitor = PerformanceMonitor()

    with settings.context(callbacks=[monitor]):
        predictor = Predict(QA)

        # Async execution
        result = await predictor.aforward(question="What is async programming?")
        print(f"\n📝 Answer: {result.answer}")

    monitor.report()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n🎯 UDSpy Callback System Examples")
    print("=" * 80)
    print("This demonstrates various callback patterns for telemetry and monitoring.")
    print("=" * 80)

    # Run synchronous examples
    example_basic_logging()
    example_performance_monitoring()
    example_cost_tracking()
    example_multiple_callbacks()
    example_per_instance_callbacks()

    # Run async example
    asyncio.run(example_async_callbacks())

    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
