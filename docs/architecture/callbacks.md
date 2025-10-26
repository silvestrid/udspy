# Callback System

The callback system provides telemetry and monitoring capabilities for tracking LLM calls, module executions, and tool invocations. It's designed to be compatible with DSPy callbacks, enabling integration with observability tools like Opik, MLflow, and other platforms that support DSPy.

## Overview

Callbacks allow you to hook into the execution flow of udspy modules and capture events at key points:

- **Module execution**: When modules (Predict, ChainOfThought, ReAct) start and complete
- **LLM calls**: When OpenAI API calls are made and receive responses
- **Tool invocations**: When tools are called and return results

This enables use cases like:
- Logging and debugging
- Performance monitoring
- Cost tracking
- Experiment tracking (MLflow, W&B)
- Observability platforms (Opik, Langfuse)

## Core Components

### BaseCallback

The base class for all callback handlers. Subclass this and implement the handlers you need:

```python
from udspy import BaseCallback

class LoggingCallback(BaseCallback):
    def on_module_start(self, call_id, instance, inputs):
        """Called when a module's forward() method starts."""
        print(f"Module {type(instance).__name__} started with inputs: {inputs}")

    def on_module_end(self, call_id, outputs, exception):
        """Called when a module's forward() method completes."""
        if exception:
            print(f"Module failed: {exception}")
        else:
            print(f"Module completed with outputs: {outputs}")

    def on_lm_start(self, call_id, instance, inputs):
        """Called when an LLM API call starts."""
        print(f"LLM call started with model: {inputs.get('model')}")

    def on_lm_end(self, call_id, outputs, exception):
        """Called when an LLM API call completes."""
        if exception:
            print(f"LLM call failed: {exception}")
        else:
            print(f"LLM call completed")

    def on_tool_start(self, call_id, instance, inputs):
        """Called when a tool is invoked."""
        print(f"Tool {instance.name} called with: {inputs}")

    def on_tool_end(self, call_id, outputs, exception):
        """Called when a tool invocation completes."""
        if exception:
            print(f"Tool failed: {exception}")
        else:
            print(f"Tool returned: {outputs}")
```

### Callback Handler Parameters

All callback handlers receive consistent parameters:

- **call_id** (str): Unique identifier for this execution (useful for tracing nested calls)
- **instance** (Any): The module or tool instance being executed (only in `_start` handlers)
- **inputs** (dict): Input parameters (only in `_start` handlers)
- **outputs** (Any | None): Execution results (only in `_end` handlers)
- **exception** (Exception | None): Exception if execution failed (only in `_end` handlers)

### with_callbacks Decorator

The `@with_callbacks` decorator is applied to module and tool methods to enable callback execution. It:

1. Retrieves active callbacks (global + instance-level)
2. Generates a unique call_id
3. Calls `on_*_start` handlers before execution
4. Executes the wrapped method
5. Calls `on_*_end` handlers after execution (even if exception occurs)

The decorator handles both sync and async methods automatically.

## Configuration

### Global Callbacks

Configure callbacks globally via `settings.configure()`:

```python
import udspy

callback = LoggingCallback()
udspy.settings.configure(
    api_key="sk-...",
    model="gpt-4o-mini",
    callbacks=[callback]  # Applied to all modules and tools
)
```

### Per-Module Callbacks

Configure callbacks for specific module instances:

```python
from udspy import Predict, Signature, InputField, OutputField

class QA(Signature):
    question: str = InputField()
    answer: str = OutputField()

# This callback only applies to this predictor instance
predictor = Predict(QA, callbacks=[LoggingCallback()])
```

### Context-Specific Callbacks

Use temporary callbacks within a context:

```python
# Temporarily override callbacks for specific operations
with udspy.settings.context(callbacks=[DebugCallback()]):
    result = predictor(question="...")
```

### Combining Callbacks

Callbacks are combined from multiple sources:

```python
# Global callbacks + instance callbacks are all executed
udspy.settings.configure(callbacks=[GlobalCallback()])
predictor = Predict(QA, callbacks=[InstanceCallback()])

# Both GlobalCallback and InstanceCallback will be invoked
result = predictor(question="...")
```

## Callback Execution Flow

### Module Execution

```
1. User calls predictor(question="...")
2. @with_callbacks on aexecute() is triggered
3. on_module_start(call_id, predictor, {"question": "...", "stream": False, ...})
4. Module executes (may call LLM and tools internally)
5. on_module_end(call_id, outputs=Prediction(...), exception=None)
6. Return result to user
```

### LLM Calls

```
1. Module calls OpenAI API
2. on_lm_start(call_id, module, {"messages": [...], "model": "...", ...})
3. API request is made
4. on_lm_end(call_id, outputs={"response": {...}}, exception=None)
5. Response is processed
```

### Tool Invocations

```
1. Module calls tool.acall(...)
2. @with_callbacks on acall() is triggered
3. on_tool_start(call_id, tool, {"query": "..."})
4. Tool function executes
5. on_tool_end(call_id, outputs="...", exception=None)
6. Return result to module
```

### Nested Calls

The callback system tracks nested calls using `ACTIVE_CALL_ID` ContextVar:

```
Module.aexecute()  -> call_id_1
├─ on_module_start(call_id_1)
├─ LLM call        -> call_id_2 (parent: call_id_1)
│  ├─ on_lm_start(call_id_2)
│  └─ on_lm_end(call_id_2)
├─ Tool call       -> call_id_3 (parent: call_id_1)
│  ├─ on_tool_start(call_id_3)
│  └─ on_tool_end(call_id_3)
└─ on_module_end(call_id_1)
```

## Error Handling

Callbacks are designed to be non-invasive:

- **Exceptions in callbacks are caught and logged** - they don't break execution
- **Failed callbacks don't affect module behavior** - other callbacks still run
- **Logging warnings are emitted** when callbacks fail

```python
class FaultyCallback(BaseCallback):
    def on_module_start(self, call_id, instance, inputs):
        raise ValueError("Oops!")  # This won't break the module

# Module still executes normally, warning is logged
predictor = Predict(QA, callbacks=[FaultyCallback()])
result = predictor(question="...")  # Works fine
```

## DSPy Compatibility

The callback interface is designed to be compatible with DSPy's callback system. This means:

1. **Same handler names**: `on_module_start`, `on_module_end`, `on_lm_start`, `on_lm_end`, `on_tool_start`, `on_tool_end`
2. **Same parameter structure**: `call_id`, `instance`, `inputs`, `outputs`, `exception`
3. **Same execution model**: Callbacks are invoked before/after operations

Tools like Opik and MLflow that provide DSPy callbacks will work with udspy:

```python
# Example with Opik (hypothetical - check Opik docs for actual API)
from opik import OpikCallback

udspy.settings.configure(
    api_key="sk-...",
    callbacks=[OpikCallback(project="my-project")]
)

# All LLM calls and module executions are now tracked in Opik
```

## Performance Considerations

- **Minimal overhead**: Callbacks only add overhead if configured
- **No overhead when disabled**: If no callbacks are set, decorator short-circuits immediately
- **Async-friendly**: Callbacks don't block async execution
- **Thread-safe**: Uses ContextVar for proper isolation

## Best Practices

### 1. Use Global Callbacks for Cross-Cutting Concerns

```python
# Logging, metrics, cost tracking
udspy.settings.configure(callbacks=[
    LoggingCallback(),
    MetricsCallback(),
    CostTracker()
])
```

### 2. Use Instance Callbacks for Specific Monitoring

```python
# Monitor only critical paths
critical_predictor = Predict(ImportantTask, callbacks=[AlertCallback()])
```

### 3. Use Context Callbacks for Debugging

```python
# Enable verbose logging only when debugging
with udspy.settings.context(callbacks=[VerboseDebugCallback()]):
    result = complex_operation()
```

### 4. Implement Selective Logging

```python
class SelectiveCallback(BaseCallback):
    def on_lm_start(self, call_id, instance, inputs):
        # Only log expensive models
        if inputs.get("model") == "gpt-4":
            logger.info(f"Expensive model call: {call_id}")
```

### 5. Track Costs

```python
class CostTracker(BaseCallback):
    def __init__(self):
        self.total_cost = 0.0

    def on_lm_end(self, call_id, outputs, exception):
        if outputs and "response" in outputs:
            # Calculate cost based on tokens
            # (This is simplified - real implementation would parse response)
            self.total_cost += 0.0001  # Example cost
```

## Common Patterns

### Request/Response Logging

```python
class RequestResponseLogger(BaseCallback):
    def on_module_start(self, call_id, instance, inputs):
        logger.info(f"[{call_id}] Request: {inputs}")

    def on_module_end(self, call_id, outputs, exception):
        logger.info(f"[{call_id}] Response: {outputs}")
```

### Performance Monitoring

```python
import time

class PerformanceMonitor(BaseCallback):
    def __init__(self):
        self.start_times = {}

    def on_module_start(self, call_id, instance, inputs):
        self.start_times[call_id] = time.time()

    def on_module_end(self, call_id, outputs, exception):
        duration = time.time() - self.start_times.pop(call_id, time.time())
        logger.info(f"Module took {duration:.2f}s")
```

### Error Tracking

```python
class ErrorTracker(BaseCallback):
    def __init__(self):
        self.errors = []

    def on_module_end(self, call_id, outputs, exception):
        if exception:
            self.errors.append({
                "call_id": call_id,
                "error": str(exception),
                "timestamp": time.time()
            })
```

## Migration from DSPy

If you're using DSPy callbacks, migration is straightforward:

```python
# DSPy code
import dspy
dspy.settings.configure(callbacks=[MyCallback()])

# udspy code - exactly the same!
import udspy
udspy.settings.configure(callbacks=[MyCallback()])
```

The callback interface is identical, so existing DSPy callbacks should work without modification.
