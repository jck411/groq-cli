"""Static model registry for Groq.

Groq doesn't have a dynamic model discovery API like Gemini,
so we maintain a static list of known models.
"""

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Metadata for a Groq model."""
    name: str
    display_name: str
    description: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool = True


# Known Groq models (updated Dec 2024)
MODELS: dict[str, ModelInfo] = {
    # Llama 3.3
    "llama-3.3-70b-versatile": ModelInfo(
        name="llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B Versatile",
        description="Latest Llama 3.3, best for complex tasks",
        context_window=128000,
        max_output_tokens=32768,
    ),
    # Llama 3.1 variants
    "llama-3.1-70b-versatile": ModelInfo(
        name="llama-3.1-70b-versatile",
        display_name="Llama 3.1 70B Versatile",
        description="High capability, versatile",
        context_window=128000,
        max_output_tokens=32768,
    ),
    "llama-3.1-8b-instant": ModelInfo(
        name="llama-3.1-8b-instant",
        display_name="Llama 3.1 8B Instant",
        description="Fast responses, good for simple tasks",
        context_window=128000,
        max_output_tokens=8192,
    ),
    # Llama 3 Guard
    "llama-guard-3-8b": ModelInfo(
        name="llama-guard-3-8b",
        display_name="Llama Guard 3 8B",
        description="Content moderation model",
        context_window=8192,
        max_output_tokens=8192,
        supports_tools=False,
    ),
    # Mixtral
    "mixtral-8x7b-32768": ModelInfo(
        name="mixtral-8x7b-32768",
        display_name="Mixtral 8x7B",
        description="Mixtral MoE, good balance of speed/quality",
        context_window=32768,
        max_output_tokens=32768,
    ),
    # Gemma
    "gemma2-9b-it": ModelInfo(
        name="gemma2-9b-it",
        display_name="Gemma 2 9B",
        description="Google's Gemma 2, instruction-tuned",
        context_window=8192,
        max_output_tokens=8192,
    ),
    # Whisper (transcription only, not for chat)
    # "whisper-large-v3": excluded - not a chat model
}

# Default model
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Recommended models for display
RECOMMENDED = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]


def get_model(name: str) -> ModelInfo | None:
    """Get model info by name."""
    return MODELS.get(name)


def list_models() -> list[ModelInfo]:
    """Get all available models."""
    return list(MODELS.values())


def list_names() -> list[str]:
    """Get all model names."""
    return list(MODELS.keys())


def is_valid_model(name: str) -> bool:
    """Check if a model name is valid."""
    return name in MODELS


def get_recommended() -> list[str]:
    """Get recommended model names."""
    return RECOMMENDED
