# Groq CLI

A lightweight terminal chat client for Groq API with MCP (Model Context Protocol) tool support.

## Features

- Direct connection to Groq API (OpenAI-compatible)
- Fast inference with Llama 3.3, Mixtral, Gemma models
- MCP tool server support (shared with gemini-cli and shell-chat)
- Shared system prompt from Backend CLI settings
- Streaming responses
- Quake terminal integration

## Installation

```bash
cd /home/human/REPOS/groq-cli
uv venv
uv pip install -e .
```

## Usage

```bash
# Start with default model (llama-3.3-70b-versatile)
groq-chat

# Use specific model
groq-chat --model llama-3.1-8b-instant

# Custom MCP config
groq-chat --mcp-config ./servers.json
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/clear` | Clear conversation |
| `/models [filter]` | List available models |
| `/model` | Show current model |
| `/model <name>` | Switch model |
| `/system` | Show system prompt |
| `/system <prompt>` | Set system prompt |
| `/config` | Show all settings |
| `/set <key> <val>` | Change a setting |
| `/save` | Save config to disk |
| `/mcp` | List/toggle MCP servers |
| `/mode` | Switch quake terminal mode |
| `/quit` | Exit |

## Shortcuts

- `Ctrl+C` - Cancel current operation
- `Ctrl+D` - Exit groq-chat
