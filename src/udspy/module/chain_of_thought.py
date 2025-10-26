"""Chain of Thought reasoning module."""

from collections.abc import AsyncGenerator
from typing import Any

from udspy.adapter import ChatAdapter
from udspy.callback import with_callbacks
from udspy.module.base import Module
from udspy.module.predict import Predict
from udspy.signature import Signature, make_signature
from udspy.streaming import StreamEvent
from udspy.tool import Tool


class ChainOfThought(Module):
    """Chain of Thought reasoning module.

    Automatically adds a reasoning step before generating outputs.
    This encourages the LLM to think step-by-step, improving answer quality.

    Example:
        ```python
        class QA(Signature):
            '''Answer questions.'''
            question: str = InputField()
            answer: str = OutputField()

        # Creates predictor with automatic reasoning
        predictor = ChainOfThought(QA)
        result = predictor(question="What is 2+2?")

        print(result.reasoning)  # "Let's think step by step..."
        print(result.answer)     # "4"
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        *,
        reasoning_description: str = "Step-by-step reasoning process",
        model: str | None = None,
        tools: list[Tool] | None = None,
        adapter: ChatAdapter | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize a Chain of Thought module.

        Args:
            signature: Signature defining inputs and final outputs, or a string in
                      format "inputs -> outputs" (e.g., "question -> answer")
            reasoning_description: Description for the reasoning field
            model: Model name (overrides global default)
            tools: List of Pydantic tool models
            adapter: Custom adapter
            callbacks: Optional list of callback handlers for this module instance
            **kwargs: Additional arguments for chat completion
        """
        super().__init__(callbacks=callbacks)

        # Convert string signature to Signature class
        if isinstance(signature, str):
            signature = Signature.from_string(signature)

        self.original_signature = signature

        # Create extended signature with reasoning field
        input_fields = {
            name: field.annotation for name, field in signature.get_input_fields().items()
        }
        output_fields = {
            name: field.annotation for name, field in signature.get_output_fields().items()
        }

        # Prepend reasoning to outputs
        extended_outputs = {"reasoning": str, **output_fields}

        # Create new signature with reasoning
        extended_signature = make_signature(
            input_fields,  # type: ignore[arg-type]
            extended_outputs,  # type: ignore[arg-type]
            signature.get_instructions(),
        )

        # Override reasoning field description
        extended_signature.model_fields["reasoning"].description = reasoning_description

        # Create predictor with extended signature
        self.predict = Predict(
            extended_signature,
            model=model,
            tools=tools,
            adapter=adapter,
            callbacks=callbacks,
            **kwargs,
        )

    @with_callbacks
    async def aexecute(
        self, *, stream: bool = False, **inputs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute chain of thought prediction.

        Delegates to the wrapped Predict module's aexecute method.

        Args:
            stream: If True, request streaming from LLM provider
            **inputs: Input values matching the signature's input fields

        Returns:
            Prediction with reasoning and other output fields
        """
        return await self.predict.aexecute(stream=stream, **inputs)
