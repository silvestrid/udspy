# udspy

[![PyPI version](https://img.shields.io/pypi/v/udspy.svg)](https://pypi.org/project/udspy/)
[![Python versions](https://img.shields.io/pypi/pyversions/udspy.svg)](https://pypi.org/project/udspy/)
[![Tests](https://github.com/baserow/udspy/actions/workflows/test.yml/badge.svg)](https://github.com/baserow/udspy/actions/workflows/test.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://baserow.github.io/udspy)
[![codecov](https://codecov.io/gh/baserow/udspy/branch/main/graph/badge.svg)](https://codecov.io/gh/baserow/udspy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight DSPy-inspired library optimized for resource-constrained environments, with native OpenAI tool calling, human-in-the-loop workflows, and conversation history.

**Topics:** `python` `openai` `llm` `dspy` `pydantic` `async` `ai-framework` `function-calling` `tool-calling` `streaming` `conversational-ai` `prompt-engineering` `type-hints` `pytest` `chatbot` `agent` `human-in-the-loop`

## About This Project

This project is inspired by **[DSPy](https://github.com/stanfordnlp/dspy)**'s elegant design and core abstractions (Signatures, Modules, Predictions).

**udspy** addresses a specific use case: **resource-constrained environments**. DSPy's dependency on LiteLLM (which requires ~200MB of memory when loaded) makes it challenging to use in contexts with limited resources, such as:
- Serverless functions with memory limits
- Edge deployments
- Embedded systems
- Cost-sensitive cloud environments

**udspy** provides:
- **Minimal footprint**: Uses the OpenAI library directly (~10MB vs ~200MB)
- **OpenAI-compatible providers**: Works with any provider compatible with OpenAI's API (OpenAI, Azure OpenAI, Together AI, Groq, etc.)
- **Additional features** for common patterns:
  - Human-in-the-loop workflows with state management
  - Automatic tool calling with multi-turn conversations
  - Built-in conversation history management
  - Confirmation system for user approval of critical operations

If resource constraints aren't a concern for your use case, consider [DSPy](https://github.com/stanfordnlp/dspy) for a more feature-complete solution.

## Features

- **Pydantic-based Signatures**: Define inputs, outputs, and tools using Pydantic models
- **Human-in-the-Loop Workflows**: Built-in confirmation system for user approval, clarification, and feedback
  - `@confirm_first` decorator for requiring confirmation before execution
  - Thread-safe and asyncio task-safe state management
  - Support for approval, rejection, argument modification, and feedback
- **Automatic Tool Calling**: Use `@tool` decorator for automatic tool execution with multi-turn conversations
- **ReAct Agent**: Reasoning and acting agent with tool calling and self-reflection
- **Conversation History**: Built-in `History` class for managing multi-turn conversations
- **Optional Tool Execution**: Control whether tools execute automatically or return for manual handling
- **Module Abstraction**: Compose LLM calls with reusable modules (`Predict`, `ChainOfThought`, `ReAct`)
- **Streaming Support**: Stream reasoning and output fields incrementally with async generators
- **Minimal Dependencies**: Only requires `openai` and `pydantic` (~10MB total footprint)

## Installation

### For Development

```bash
# Clone the repository
git clone https://github.com/baserow/udspy
cd udspy

# Install dependencies and package in editable mode
uv sync
uv pip install -e .

# Or with pip
pip install -e .
```

### For Users

```bash
# When published to PyPI
pip install udspy

# Or with uv
uv pip install udspy
```

## Quick Start

### Basic Usage

```python
import udspy
from udspy import Signature, InputField, OutputField, Predict

# Configure OpenAI client
udspy.settings.configure(api_key="your-api-key", model="gpt-4o-mini")

# Define a signature
class QA(Signature):
    """Answer questions concisely."""
    question: str = InputField()
    answer: str = OutputField()

# Create and use a predictor
predictor = Predict(QA)
result = predictor(question="What is the capital of France?")
print(result.answer)
```

### With Conversation History

```python
from udspy import History

predictor = Predict(QA)
history = History()

# Multi-turn conversation
result = predictor(question="What is Python?", history=history)
print(result.answer)

result = predictor(question="What are its main features?", history=history)
print(result.answer)  # Context from previous turn is maintained
```

### With Automatic Tool Calling

```python
from udspy import tool
from pydantic import Field

@tool(name="Calculator", description="Perform arithmetic operations")
def calculator(
    operation: str = Field(description="add, subtract, multiply, divide"),
    a: float = Field(description="First number"),
    b: float = Field(description="Second number"),
) -> float:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
    return ops[operation]

predictor = Predict(QA, tools=[calculator])
result = predictor(question="What is 157 times 234?")
print(result.answer)  # Tools are automatically executed
```

### With Human-in-the-Loop

```python
from udspy import ReAct, HumanInTheLoopRequired, tool
from pydantic import Field
import os

@tool(
    name="delete_file",
    description="Delete a file",
    require_confirmation=True  # Requires user confirmation
)
def delete_file(path: str = Field(description="File path")) -> str:
    os.remove(path)
    return f"Deleted {path}"

class FileTask(Signature):
    """Perform file operations safely."""
    request: str = InputField()
    result: str = OutputField()

agent = ReAct(FileTask, tools=[delete_file])

try:
    result = agent(request="Delete /tmp/old_data.txt")
except HumanInTheLoopRequired as e:
    print(f"Agent asks: {e.question}")
    # User confirms: "yes", "no", or provides feedback
    result = agent.resume("yes", e)
    print(result.result)
```

## Development

```bash
# Install dependencies and package in editable mode
just install
uv pip install -e .

# Run tests
just test

# Run linter
just lint

# Format code
just fmt

# Type check
just typecheck

# Run all checks
just check

# Build docs
just docs-serve
```

## Documentation

Full documentation is available at [baserow.github.io/udspy](https://baserow.github.io/udspy)

Or browse locally:
- [Architecture](docs/architecture/overview.md)
- [Examples](docs/examples/)
- [API Reference](docs/api/)

### Building Documentation

```bash
# Install mkdocs dependencies
pip install mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build static site
mkdocs build
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Releases

Releases are automated via GitHub Actions:

1. Update version in `pyproject.toml` and `src/udspy/__init__.py`
2. Commit and tag: `git tag v0.x.x && git push --tags`
3. GitHub Actions will build, test, and publish to PyPI
4. Documentation will be deployed to GitHub Pages

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed release instructions.

## License

MIT
