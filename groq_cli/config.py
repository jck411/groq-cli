"""Configuration management for Groq CLI.

Supports persistent config stored in ~/.config/groq-cli/config.json
with hot reload - settings take effect immediately when changed.

System prompt is shared with shell-chat and gemini-chat from Backend's CLI settings.
"""

import json
from pathlib import Path
from typing import Any

from .models import DEFAULT_MODEL, is_valid_model, get_recommended


# Backend's shared CLI settings (system_prompt shared with shell-chat/gemini-chat)
BACKEND_CLI_SETTINGS = Path("/home/human/REPOS/Backend_FastAPI/src/backend/data/clients/cli/llm.json")

# Default configuration values
DEFAULTS = {
    "model": DEFAULT_MODEL,
    "system_prompt": "",
    "temperature": 0.7,
    "max_tokens": 4096,
    "mcp_servers": {},  # server_name -> enabled (bool)
    "debug": False,  # Show verbose tool call logs
}


class Config:
    """Manages persistent configuration with hot reload support.
    
    All settings take effect immediately when changed via set().
    Use save() to persist changes to disk.
    System prompt is loaded from Backend's shared CLI settings.
    """

    def __init__(self, config_dir: Path | None = None):
        """Initialize config, loading from disk if available."""
        if config_dir is None:
            config_dir = Path.home() / ".config" / "groq-cli"
        
        self.config_dir = config_dir
        self.config_file = config_dir / "config.json"
        self._config: dict[str, Any] = DEFAULTS.copy()
        self._load()
        self._load_backend_settings()

    def _load(self) -> bool:
        """Load config from disk, merging with defaults."""
        if not self.config_file.exists():
            return False
        
        try:
            with open(self.config_file) as f:
                loaded = json.load(f)
            
            # Merge with defaults (loaded values override defaults)
            for key, value in loaded.items():
                if key in DEFAULTS:
                    self._config[key] = value
            
            return True
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load config: {e}")
            return False

    def _load_backend_settings(self) -> bool:
        """Load system_prompt from Backend's shared CLI settings."""
        if not BACKEND_CLI_SETTINGS.exists():
            return False
        
        try:
            with open(BACKEND_CLI_SETTINGS) as f:
                backend = json.load(f)
            
            # Use Backend's system prompt (shared with shell-chat/gemini-chat)
            if "system_prompt" in backend and backend["system_prompt"]:
                self._config["system_prompt"] = backend["system_prompt"]
            
            return True
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not load Backend CLI settings: {e}")
            return False

    def save(self) -> bool:
        """Save current config to disk."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self._config, f, indent=2)
            return True
        except OSError as e:
            print(f"Error saving config: {e}")
            return False

    def reload(self) -> bool:
        """Reload config from disk (hot reload)."""
        return self._load()

    # Property accessors - settings take effect immediately
    @property
    def model(self) -> str:
        return self._config["model"]

    @model.setter
    def model(self, value: str) -> None:
        # Validate against known models
        if is_valid_model(value):
            self._config["model"] = value
        else:
            suggestions = get_recommended()[:3]
            raise ValueError(
                f"Unknown model: {value}. Try: {', '.join(suggestions)} "
                f"(use /models to see all)"
            )

    @property
    def system_prompt(self) -> str:
        return self._config["system_prompt"]

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._config["system_prompt"] = value

    @property
    def temperature(self) -> float:
        return self._config["temperature"]

    @temperature.setter
    def temperature(self, value: float) -> None:
        if 0.0 <= value <= 2.0:
            self._config["temperature"] = value
        else:
            raise ValueError("Temperature must be between 0.0 and 2.0")

    @property
    def max_tokens(self) -> int:
        return self._config["max_tokens"]

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        if value > 0:
            self._config["max_tokens"] = value
        else:
            raise ValueError("max_tokens must be positive")

    @property
    def mcp_servers(self) -> dict[str, bool]:
        return self._config["mcp_servers"]

    def set_mcp_server(self, name: str, enabled: bool) -> None:
        """Enable or disable an MCP server."""
        self._config["mcp_servers"][name] = enabled

    def is_mcp_server_enabled(self, name: str) -> bool:
        """Check if an MCP server is enabled (default: True if not specified)."""
        return self._config["mcp_servers"].get(name, True)

    @property
    def debug(self) -> bool:
        return self._config.get("debug", False)

    @debug.setter
    def debug(self, value: bool) -> None:
        self._config["debug"] = bool(value)

    # Generic set method for runtime commands
    def set(self, key: str, value: Any) -> None:
        """Set a config value by key name. Raises ValueError for invalid values."""
        if key == "model":
            self.model = value
        elif key == "system_prompt" or key == "system":
            self.system_prompt = str(value)
        elif key == "temperature" or key == "temp":
            self.temperature = float(value)
        elif key == "max_tokens":
            self.max_tokens = int(value)
        else:
            raise ValueError(f"Unknown setting: {key}")

    def get(self, key: str) -> Any:
        """Get a config value by key name."""
        if key in self._config:
            return self._config[key]
        raise ValueError(f"Unknown setting: {key}")

    def to_dict(self) -> dict[str, Any]:
        """Return config as a dictionary."""
        return self._config.copy()

    def __repr__(self) -> str:
        return f"Config({self._config})"
