"""udspy: A minimal DSPy-inspired library with native OpenAI tool calling."""

from udspy.adapter import ChatAdapter
from udspy.callback import ACTIVE_CALL_ID, BaseCallback, with_callbacks
from udspy.confirmation import (
    ConfirmationRejected,
    ConfirmationRequired,
    ResumeState,
    ToolCall,
    confirm_first,
    get_confirmation_status,
    respond_to_confirmation,
)
from udspy.history import History
from udspy.module import (
    ChainOfThought,
    Module,
    Predict,
    Prediction,
    ReAct,
)
from udspy.settings import settings
from udspy.signature import InputField, OutputField, Signature, make_signature
from udspy.streaming import StreamChunk, StreamEvent, emit_event
from udspy.tool import Tool, tool

__version__ = "0.1.2"

__all__ = [
    # Settings
    "settings",
    # Signatures
    "Signature",
    "InputField",
    "OutputField",
    "make_signature",
    # Modules
    "Module",
    "Predict",
    "Prediction",
    "ChainOfThought",
    "ReAct",
    # Callbacks
    "BaseCallback",
    "with_callbacks",
    "ACTIVE_CALL_ID",
    # Confirmation
    "ConfirmationRequired",
    "ConfirmationRejected",
    "ResumeState",
    "confirm_first",
    "ToolCall",
    "get_confirmation_status",
    "respond_to_confirmation",
    # Adapter
    "ChatAdapter",
    # History
    "History",
    # Streaming
    "StreamEvent",
    "StreamChunk",
    "emit_event",
    # Tools
    "Tool",
    "tool",
]
