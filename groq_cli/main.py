#!/usr/bin/env python3
"""Groq CLI - Terminal chat with MCP tool support.

A lightweight terminal client that talks directly to Groq API
(OpenAI-compatible) with support for MCP (Model Context Protocol) tool servers.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.style import Style
from rich.text import Text

from .config import Config
from .mcp_client import MCPClient
from .models import get_model, list_models, list_names, get_recommended, DEFAULT_MODEL

# Load environment variables
load_dotenv()

# Styles
USER_STYLE = Style(color="bright_blue", bold=True)
ASSISTANT_STYLE = Style(color="bright_green")
TOOL_STYLE = Style(color="yellow")
ERROR_STYLE = Style(color="red", bold=True)
INFO_STYLE = Style(color="cyan")

# Groq API base URL
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqChat:
    """Terminal chat client for Groq with MCP tools."""

    def __init__(
        self,
        model: str | None = None,
        mcp_config: Path | None = None,
        config: Config | None = None,
    ):
        self.console = Console()
        self.running = True

        # Load config (hot reload enabled)
        self.config = config or Config()

        # Use CLI arg model if provided, otherwise use config
        self.model_name = model or self.config.model
        self.mcp_config = mcp_config

        # Initialize Groq client (OpenAI-compatible)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.console.print(
                "[error]GROQ_API_KEY not set. Add it to .env or environment.[/error]",
                style=ERROR_STYLE,
            )
            sys.exit(1)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
        )

        # Initialize MCP client (connected async later)
        self.mcp = MCPClient()
        self.tools: list[dict[str, Any]] | None = None

        # Conversation history (OpenAI format)
        self.history: list[dict[str, Any]] = []

        # For numbered selection caching
        self._last_model_list: list | None = None
        self._last_mcp_list: list | None = None

    async def initialize(self) -> None:
        """Async initialization."""
        # Load MCP config from Backend's shared config
        if self.mcp_config:
            self.mcp.load_config(self.mcp_config)
        else:
            # Auto-load from Backend's shared mcp_servers.json
            self.mcp.load_config()

        # Connect to all configured MCP servers
        await self.mcp.connect_all()

        # Show MCP connection status in debug mode
        if self.config.debug:
            self.console.print("\n[bold yellow]═══ DEBUG: MCP Servers ═══[/bold yellow]")
            for info in self.mcp.list_servers():
                status = "[green]✓[/green]" if info["running"] else "[red]✗[/red]"
                transport = "HTTP" if info["http"] else "stdio"
                self.console.print(f"  {status} {info['name']} ({transport}) - {info['tools']} tools")
            self.console.print("[bold yellow]══════════════════════════[/bold yellow]\n")

        # Build tool definitions
        self._init_tools()

    async def shutdown(self) -> None:
        """Async cleanup."""
        await self.mcp.disconnect_all()

    def _init_tools(self) -> None:
        """Initialize tool definitions from MCP servers."""
        mcp_tools = self.mcp.get_all_tools()

        if mcp_tools:
            self.tools = mcp_tools
        else:
            self.tools = None

    def _clear_session(self) -> None:
        """Clear conversation history."""
        self.history = []
        self.console.print(
            "[info]Conversation cleared.[/info]", style=INFO_STYLE
        )

    def _show_model(self) -> None:
        """Show current model."""
        self.console.print(
            f"[info]Current model: {self.model_name}[/info]", style=INFO_STYLE
        )

    def _set_model(self, model_name: str) -> None:
        """Switch to a different model (hot reload + auto-save)."""
        try:
            self.config.model = model_name
            self.model_name = model_name
            self.history = []
            self.config.save()  # Auto-save so it persists
            self.console.print(
                f"[info]Switched to {model_name}. Conversation cleared.[/info]",
                style=INFO_STYLE,
            )
        except ValueError as e:
            self.console.print(f"[error]{e}[/error]", style=ERROR_STYLE)

    def _show_config(self) -> None:
        """Show current configuration."""
        cfg = self.config.to_dict()
        self.console.print("\n[bold]Current Configuration:[/bold]")
        self.console.print(f"  model: {cfg['model']}")
        self.console.print(f"  temperature: {cfg['temperature']}")
        self.console.print(f"  max_tokens: {cfg['max_tokens']}")
        sys_prompt = cfg['system_prompt']
        if sys_prompt:
            display = sys_prompt[:50] + "..." if len(sys_prompt) > 50 else sys_prompt
            self.console.print(f"  system_prompt: {display}")
        else:
            self.console.print("  system_prompt: [dim](not set)[/dim]")
        self.console.print()

    def _set_config(self, key: str, value: str) -> None:
        """Set a config value (hot reload)."""
        try:
            self.config.set(key, value)
            # Sync model_name if model was changed
            if key == "model":
                self.model_name = self.config.model
                self.history = []
            self.console.print(f"[info]Set {key} = {value}[/info]", style=INFO_STYLE)
        except ValueError as e:
            self.console.print(f"[error]{e}[/error]", style=ERROR_STYLE)

    def _save_config(self) -> None:
        """Save current config to disk."""
        if self.config.save():
            self.console.print(
                f"[info]Config saved to {self.config.config_file}[/info]",
                style=INFO_STYLE,
            )
        else:
            self.console.print("[error]Failed to save config[/error]", style=ERROR_STYLE)

    def _show_system_prompt(self) -> None:
        """Show current system prompt."""
        prompt = self.config.system_prompt
        if prompt:
            self.console.print(f"\n[bold]System Prompt:[/bold]\n{prompt}\n")
        else:
            self.console.print("[dim]No system prompt set. Use /system <prompt> to set one.[/dim]")

    def _set_system_prompt(self, prompt: str) -> None:
        """Set system prompt (hot reload)."""
        self.config.system_prompt = prompt
        self.console.print("[info]System prompt updated.[/info]", style=INFO_STYLE)

    def _list_mcp(self, arg: str | None = None) -> None:
        """List available MCP servers and activate by typing numbers like '124'."""
        # Discover available servers from Backend_FastAPI
        mcp_dir = Path("/home/human/REPOS/Backend_FastAPI/src/backend/mcp_servers")

        if not mcp_dir.exists():
            self.console.print(f"[error]MCP directory not found: {mcp_dir}[/error]", style=ERROR_STYLE)
            return

        # Find all *_server.py files
        available = []
        for f in sorted(mcp_dir.glob("*_server.py")):
            name = f.stem.replace("_server", "")  # e.g., "shell_control"
            available.append({"name": name, "file": f.name, "path": str(f)})

        if not available:
            self.console.print("[dim]No MCP servers found[/dim]")
            return

        # If arg provided, it's a selection like "124" meaning activate 1, 2, 4
        if arg and arg.isdigit():
            # Handle "0" as cancel
            if arg == "0":
                self._last_mcp_list = None  # Clear cache
                self.console.print("[dim]Cancelled[/dim]")
                return

            selected_indices = [int(d) - 1 for d in arg]  # Parse each digit
            selected_names = []

            for idx in selected_indices:
                if 0 <= idx < len(available):
                    selected_names.append(available[idx]["name"])

            if not selected_names:
                self.console.print("[error]No valid selections[/error]", style=ERROR_STYLE)
                return

            # Save to config
            for server in available:
                self.config.set_mcp_server(server["name"], server["name"] in selected_names)
            self.config.save()

            # Generate mcp_servers.json
            self._generate_mcp_config(available, selected_names)

            self.console.print(f"[info]Activated: {', '.join(selected_names)}[/info]", style=INFO_STYLE)
            self.console.print("[dim]Restart groq-chat to load the servers.[/dim]")
            return

        # Show numbered list
        self.console.print(f"\n[bold]Available MCP Servers ({len(available)}):[/bold]")
        for i, server in enumerate(available, 1):
            enabled = self.config.is_mcp_server_enabled(server["name"])
            status = "[green]✓[/green]" if enabled else "[dim]○[/dim]"
            self.console.print(f"  [yellow]{i}[/yellow]. {status} {server['name']}")

        self.console.print()
        self.console.print("[dim]Type digits to activate (e.g. '134'), or 0 to cancel[/dim]")
        self.console.print()

        # Cache for bare number handling
        self._last_mcp_list = available

    def _generate_mcp_config(self, available: list, selected_names: list) -> None:
        """Generate mcp_servers.json with selected servers."""
        config_path = self.config.config_dir / "mcp_servers.json"

        servers = {}
        for server in available:
            if server["name"] in selected_names:
                servers[server["name"]] = {
                    "command": "/home/human/REPOS/Backend_FastAPI/.venv/bin/python",
                    "args": ["-m", f"src.backend.mcp_servers.{server['file'][:-3]}"],
                    "cwd": "/home/human/REPOS/Backend_FastAPI",
                    "env": {
                        "HOST_PROFILE_ID": "xps13",
                        "HOST_ROOT_PATH": "/home/human/GoogleDrive/host_profiles"
                    }
                }

        mcp_config = {"servers": servers}

        try:
            with open(config_path, "w") as f:
                json.dump(mcp_config, f, indent=2)
            self.console.print(f"[dim]Generated {config_path}[/dim]")
        except Exception as e:
            self.console.print(f"[error]Failed to generate MCP config: {e}[/error]", style=ERROR_STYLE)

    def _handle_mode(self, mode: str | None = None) -> None:
        """Handle /mode command - switch quake terminal mode."""
        mode_file = Path.home() / ".config" / "quake-llm-terminal" / "default-provider"
        valid_modes = ["terminal", "openrouter", "gemini", "groq"]

        # Get current mode
        current = "terminal"
        if mode_file.exists():
            current = mode_file.read_text().strip()

        if not mode:
            # Show current mode and options
            self.console.print(f"\n[bold]Quake Terminal Mode:[/bold]")
            for m in valid_modes:
                marker = "[green]►[/green]" if m == current else " "
                desc = {
                    "terminal": "Plain shell",
                    "openrouter": "shell-chat (OpenRouter via Backend)",
                    "gemini": "gemini-chat (Gemini API)",
                    "groq": "groq-chat (Groq API)"
                }.get(m, "")
                self.console.print(f"  {marker} [cyan]{m}[/cyan] - {desc}")
            self.console.print(f"\n[dim]Usage: /mode <terminal|openrouter|gemini|groq>[/dim]")
            return

        mode = mode.lower()
        if mode not in valid_modes:
            self.console.print(f"[error]Invalid mode: {mode}[/error]", style=ERROR_STYLE)
            self.console.print(f"[dim]Use: terminal, openrouter, gemini, or groq[/dim]")
            return

        # Save new mode
        mode_file.parent.mkdir(parents=True, exist_ok=True)
        mode_file.write_text(mode)
        self.console.print(f"[info]Mode set to: {mode}[/info]", style=INFO_STYLE)

        if mode != "groq":
            # Exit and let quake terminal restart with new mode
            self.console.print("[dim]Exiting to switch mode. Press quake hotkey to reopen.[/dim]")
            self.running = False

    def _list_models(self, filter_text: str | None = None) -> None:
        """List available models with numbered selection."""
        models = list_models()

        # Check if filter_text is a number (selection)
        if filter_text and filter_text.isdigit():
            # Handle "0" as cancel
            if filter_text == "0":
                self._last_model_list = None
                self.console.print("[dim]Cancelled[/dim]")
                return

            idx = int(filter_text) - 1  # 1-indexed
            if hasattr(self, '_last_model_list') and self._last_model_list and 0 <= idx < len(self._last_model_list):
                selected = self._last_model_list[idx]
                self._set_model(selected.name)
                return
            else:
                self.console.print("[error]Invalid selection. Run /models first.[/error]", style=ERROR_STYLE)
                return

        # Filter if provided (non-numeric)
        if filter_text:
            filter_lower = filter_text.lower()
            models = [m for m in models if filter_lower in m.name.lower()]

        if not models:
            self.console.print("[dim]No models found matching filter[/dim]")
            return

        # Cache for selection
        self._last_model_list = models

        # Show recommended first
        recommended = get_recommended()

        self.console.print(f"\n[bold]Available Models ({len(models)}):[/bold]")
        for i, model in enumerate(models, 1):
            is_rec = model.name in recommended
            is_current = model.name == self.model_name
            rec_marker = " ⭐" if is_rec else ""
            current_marker = " [green]◀ current[/green]" if is_current else ""
            self.console.print(
                f"  [yellow]{i:2}[/yellow]. [cyan]{model.name}[/cyan]{rec_marker}{current_marker}"
            )
            self.console.print(
                f"      [dim]{model.description} (ctx: {model.context_window})[/dim]"
            )
        self.console.print()
        self.console.print("[dim]Type a number to switch, or 0 to cancel[/dim]")
        self.console.print()

    def _show_help(self) -> None:
        """Show help message."""
        help_text = """
[bold]Commands:[/bold]
  /help              Show this help
  /clear             Clear conversation
  /debug             Toggle debug logging
  /models [filter]   List available models
  /model             Show current model
  /model <name>      Switch model
  /system            Show system prompt
  /system <prompt>   Set system prompt
  /config            Show all settings
  /set <key> <val>   Change a setting
  /save              Save config to disk
  /mcp               List/toggle MCP servers
  /mode              Switch quake terminal mode
  /quit              Exit

[bold]Settings:[/bold]
  model, temperature, max_tokens, system

[bold]Shortcuts:[/bold]
  Ctrl+C  Cancel   Ctrl+D  Exit
"""
        self.console.print(
            Panel(help_text.strip(), title="Groq CLI Help", border_style="blue")
        )

    def _toggle_debug(self) -> None:
        """Toggle debug mode on/off."""
        self.config.debug = not self.config.debug
        self.config.save()
        status = "ON" if self.config.debug else "OFF"
        self.console.print(
            f"[info]Debug mode: {status}[/info]", style=INFO_STYLE
        )

    def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        parts = cmd.strip().split(maxsplit=1)
        if not parts:
            return False

        command = parts[0].lower()

        if command == "/help":
            self._show_help()
            return True
        elif command == "/clear":
            self._clear_session()
            return True
        elif command == "/debug":
            self._toggle_debug()
            return True
        elif command == "/quit":
            self.running = False
            return True
        elif command == "/model":
            if len(parts) > 1:
                self._set_model(parts[1])
            else:
                self._show_model()
            return True
        elif command == "/models":
            filter_text = parts[1] if len(parts) > 1 else None
            self._list_models(filter_text)
            return True
        elif command == "/mcp":
            arg = parts[1] if len(parts) > 1 else None
            self._list_mcp(arg)
            return True
        elif command == "/config":
            self._show_config()
            return True
        elif command == "/save":
            self._save_config()
            return True
        elif command == "/system":
            if len(parts) > 1:
                self._set_system_prompt(parts[1])
            else:
                self._show_system_prompt()
            return True
        elif command == "/set":
            if len(parts) > 1:
                set_parts = parts[1].split(maxsplit=1)
                if len(set_parts) >= 2:
                    self._set_config(set_parts[0], set_parts[1])
                else:
                    self.console.print("[error]Usage: /set <key> <value>[/error]", style=ERROR_STYLE)
            else:
                self.console.print("[error]Usage: /set <key> <value>[/error]", style=ERROR_STYLE)
            return True
        elif command == "/mode":
            self._handle_mode(parts[1] if len(parts) > 1 else None)
            return True

        return False

    def _handle_error(self, e: Exception, context: str = "") -> None:
        """Handle errors with full debug info when debug mode is on."""
        debug = self.config.debug

        self.console.print(f"\n[error]Error: {e}[/error]", style=ERROR_STYLE)

        if debug:
            self.console.print("\n[bold red]═══ DEBUG: Error Details ═══[/bold red]")
            self.console.print(f"[dim]Context: {context}[/dim]")
            self.console.print(f"[dim]Exception type: {type(e).__name__}[/dim]")
            self.console.print(f"[dim]Exception str: {str(e)}[/dim]")

            # Try to extract more info from the exception
            if hasattr(e, 'response'):
                resp = e.response
                self.console.print(f"[dim]Response status: {getattr(resp, 'status_code', 'N/A')}[/dim]")
                if hasattr(resp, 'text'):
                    self.console.print(f"[dim]Response body:[/dim]")
                    try:
                        body = json.loads(resp.text)
                        self.console.print(f"[dim]{json.dumps(body, indent=2)}[/dim]")
                        # Look for Groq-specific error fields
                        if 'error' in body:
                            error = body['error']
                            if 'failed_generation' in error:
                                self.console.print(f"\n[bold yellow]failed_generation:[/bold yellow]")
                                self.console.print(f"[dim]{error['failed_generation']}[/dim]")
                            if 'message' in error:
                                self.console.print(f"[dim]Error message: {error['message']}[/dim]")
                    except:
                        self.console.print(f"[dim]{resp.text}[/dim]")

            if hasattr(e, 'body'):
                self.console.print(f"[dim]Error body: {e.body}[/dim]")
                # Groq errors often have 'failed_generation' in body
                if isinstance(e.body, dict):
                    if 'failed_generation' in e.body:
                        self.console.print(f"\n[bold yellow]failed_generation:[/bold yellow]")
                        self.console.print(f"[dim]{e.body['failed_generation']}[/dim]")
                    if 'error' in e.body and isinstance(e.body['error'], dict):
                        if 'failed_generation' in e.body['error']:
                            self.console.print(f"\n[bold yellow]failed_generation:[/bold yellow]")
                            self.console.print(f"[dim]{e.body['error']['failed_generation']}[/dim]")

            # Print full traceback
            self.console.print(f"\n[dim]Traceback:[/dim]")
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            self.console.print("[bold red]════════════════════════════[/bold red]\n")

    async def _execute_tool_calls(self, tool_calls: list) -> list[dict[str, Any]]:
        """Execute tool calls from OpenAI response."""
        tool_results = []
        debug = self.config.debug

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments

            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as je:
                self.console.print(
                    f"[error]Failed to parse tool arguments for {tool_name}[/error]",
                    style=ERROR_STYLE
                )
                if debug:
                    self.console.print(f"[dim]  Raw arguments: {raw_arguments}[/dim]")
                    self.console.print(f"[dim]  JSON error: {je}[/dim]")
                arguments = {}

            self.console.print(
                Text(f"🔧 Calling {tool_name}...", style=TOOL_STYLE)
            )

            # Show arguments in debug mode
            if debug:
                args_formatted = json.dumps(arguments, indent=2)
                self.console.print(f"[dim]  Args: {args_formatted}[/dim]")

            # Execute tool via MCP (async)
            result = await self.mcp.call_tool(tool_name, arguments)

            # Show result (full in debug mode, truncated otherwise)
            if debug:
                self.console.print(f"[dim]  Result:[/dim]")
                self.console.print(f"[dim]{result}[/dim]")
            else:
                display_result = result[:200] + "..." if len(result) > 200 else result
                self.console.print(f"[dim]  → {display_result}[/dim]")

            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        return tool_results

    async def _send_message(self, message: str) -> None:
        """Send message to Groq and handle response with streaming."""
        debug = self.config.debug

        try:
            # Build messages list
            messages = []

            # Add system prompt if set
            if self.config.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.config.system_prompt
                })

            # Add conversation history
            messages.extend(self.history)

            # Add user message
            messages.append({
                "role": "user",
                "content": message
            })

            # Build request kwargs
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
            }

            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            # Debug: show full request
            if debug:
                self.console.print("\n[bold yellow]═══ DEBUG: API Request ═══[/bold yellow]")
                self.console.print(f"[dim]Model: {self.model_name}[/dim]")
                self.console.print(f"[dim]Temperature: {self.config.temperature} | Max tokens: {self.config.max_tokens}[/dim]")
                self.console.print(f"[dim]Message count: {len(messages)}[/dim]")

                # Show tool schemas (first time only, or if requested)
                if self.tools:
                    tool_names = [t.get('function', {}).get('name', '?') for t in self.tools]
                    self.console.print(f"[dim]Tools ({len(self.tools)}): {', '.join(tool_names)}[/dim]")

                self.console.print("[dim]Messages (full):[/dim]")
                for i, msg in enumerate(messages):
                    role = msg.get('role', '?')
                    content = msg.get('content', '')
                    tool_calls = msg.get('tool_calls', [])

                    if content:
                        # Show full content, properly formatted
                        self.console.print(f"  [dim]{i}. [{role}]:[/dim]")
                        for line in str(content)[:2000].split('\n'):  # Cap at 2000 chars for sanity
                            self.console.print(f"      [dim]{line}[/dim]")
                        if len(str(content)) > 2000:
                            self.console.print(f"      [dim]... (truncated, {len(str(content))} chars total)[/dim]")
                    elif tool_calls:
                        self.console.print(f"  [dim]{i}. [{role}]: (tool_calls: {len(tool_calls)})[/dim]")
                    else:
                        self.console.print(f"  [dim]{i}. [{role}]: (no content)[/dim]")
                self.console.print("[bold yellow]══════════════════════════[/bold yellow]\n")

            # Send request with streaming
            stream = await self.client.chat.completions.create(**kwargs)

            # Collect streamed response
            full_content = ""
            tool_calls_data: dict[int, dict] = {}  # index -> {id, name, arguments}
            chunk_count = 0

            async for chunk in stream:
                chunk_count += 1

                # Debug: show raw chunk data
                if debug:
                    self.console.print(f"[dim]  [chunk {chunk_count}]: {chunk}[/dim]")

                # Check for errors in chunk (Groq sometimes returns errors here)
                if hasattr(chunk, 'error') and chunk.error:
                    self.console.print(f"\n[error]API Error in chunk: {chunk.error}[/error]", style=ERROR_STYLE)
                    if debug:
                        self.console.print(f"[dim]Full error chunk: {chunk}[/dim]")
                    continue

                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Check for finish_reason that indicates issues
                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
                if debug and finish_reason:
                    self.console.print(f"[dim]  [finish_reason]: {finish_reason}[/dim]")

                # Handle text content
                if delta.content:
                    full_content += delta.content
                    self.console.print(delta.content, end="")

                # Handle tool calls (streamed)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_data[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_data[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_data[idx]["arguments"] += tc.function.arguments

            self.console.print()  # Newline after streaming

            if debug:
                self.console.print(f"[dim]Total chunks received: {chunk_count}[/dim]")

            # Update history with user message
            self.history.append({"role": "user", "content": message})

            # Handle tool calls if any
            if tool_calls_data:
                if debug:
                    self.console.print("\n[bold yellow]═══ DEBUG: Tool Calls ═══[/bold yellow]")
                    for idx, tc_data in tool_calls_data.items():
                        self.console.print(f"[dim]Tool {idx}:[/dim]")
                        self.console.print(f"[dim]  ID: {tc_data['id']}[/dim]")
                        self.console.print(f"[dim]  Name: {tc_data['name']}[/dim]")
                        self.console.print(f"[dim]  Arguments (raw): {tc_data['arguments']}[/dim]")
                    self.console.print("[bold yellow]═════════════════════════[/bold yellow]\n")

                # Build tool_calls list for history
                assistant_tool_calls = []
                for idx in sorted(tool_calls_data.keys()):
                    tc_data = tool_calls_data[idx]
                    assistant_tool_calls.append({
                        "id": tc_data["id"],
                        "type": "function",
                        "function": {
                            "name": tc_data["name"],
                            "arguments": tc_data["arguments"]
                        }
                    })

                # Add assistant message with tool calls
                self.history.append({
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": assistant_tool_calls
                })

                # Create mock tool_call objects for execution
                class MockFunction:
                    def __init__(self, name: str, arguments: str):
                        self.name = name
                        self.arguments = arguments

                class MockToolCall:
                    def __init__(self, id: str, function: MockFunction):
                        self.id = id
                        self.function = function

                mock_calls = [
                    MockToolCall(tc["id"], MockFunction(tc["function"]["name"], tc["function"]["arguments"]))
                    for tc in assistant_tool_calls
                ]

                # Execute tool calls
                tool_results = await self._execute_tool_calls(mock_calls)

                # Add tool results to history
                self.history.extend(tool_results)

                # Continue conversation with tool results
                await self._continue_after_tools()
            else:
                # No tool calls, just add assistant response
                if full_content:
                    self.history.append({
                        "role": "assistant",
                        "content": full_content
                    })

        except Exception as e:
            self._handle_error(e, "send_message")

    async def _continue_after_tools(self) -> None:
        """Continue conversation after tool execution."""
        debug = self.config.debug

        try:
            # Build messages with all history including tool results
            messages = []

            if self.config.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.config.system_prompt
                })

            messages.extend(self.history)

            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
            }

            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            if debug:
                self.console.print("\n[bold yellow]═══ DEBUG: Continue After Tools ═══[/bold yellow]")
                self.console.print(f"[dim]Message count: {len(messages)}[/dim]")
                self.console.print("[bold yellow]════════════════════════════════════[/bold yellow]\n")

            stream = await self.client.chat.completions.create(**kwargs)

            full_content = ""
            tool_calls_data: dict[int, dict] = {}
            chunk_count = 0

            async for chunk in stream:
                chunk_count += 1

                if debug:
                    self.console.print(f"[dim]  [chunk {chunk_count}]: {chunk}[/dim]")

                # Check for errors in chunk
                if hasattr(chunk, 'error') and chunk.error:
                    self.console.print(f"\n[error]API Error in chunk: {chunk.error}[/error]", style=ERROR_STYLE)
                    continue

                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                # Check for finish_reason
                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
                if debug and finish_reason:
                    self.console.print(f"[dim]  [finish_reason]: {finish_reason}[/dim]")

                if delta.content:
                    full_content += delta.content
                    self.console.print(delta.content, end="")

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_data[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_data[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_data[idx]["arguments"] += tc.function.arguments

            self.console.print()

            if debug:
                self.console.print(f"[dim]Total chunks received: {chunk_count}[/dim]")

            # Handle more tool calls if needed (recursive)
            if tool_calls_data:
                if debug:
                    self.console.print("\n[bold yellow]═══ DEBUG: More Tool Calls ═══[/bold yellow]")
                    for idx, tc_data in tool_calls_data.items():
                        self.console.print(f"[dim]Tool {idx}:[/dim]")
                        self.console.print(f"[dim]  ID: {tc_data['id']}[/dim]")
                        self.console.print(f"[dim]  Name: {tc_data['name']}[/dim]")
                        self.console.print(f"[dim]  Arguments (raw): {tc_data['arguments']}[/dim]")
                    self.console.print("[bold yellow]══════════════════════════════[/bold yellow]\n")

                assistant_tool_calls = []
                for idx in sorted(tool_calls_data.keys()):
                    tc_data = tool_calls_data[idx]
                    assistant_tool_calls.append({
                        "id": tc_data["id"],
                        "type": "function",
                        "function": {
                            "name": tc_data["name"],
                            "arguments": tc_data["arguments"]
                        }
                    })

                self.history.append({
                    "role": "assistant",
                    "content": full_content or None,
                    "tool_calls": assistant_tool_calls
                })

                class MockFunction:
                    def __init__(self, name: str, arguments: str):
                        self.name = name
                        self.arguments = arguments

                class MockToolCall:
                    def __init__(self, id: str, function: MockFunction):
                        self.id = id
                        self.function = function

                mock_calls = [
                    MockToolCall(tc["id"], MockFunction(tc["function"]["name"], tc["function"]["arguments"]))
                    for tc in assistant_tool_calls
                ]

                tool_results = await self._execute_tool_calls(mock_calls)
                self.history.extend(tool_results)

                # Recursive call (with depth limit in practice)
                await self._continue_after_tools()
            else:
                if full_content:
                    self.history.append({
                        "role": "assistant",
                        "content": full_content
                    })

        except Exception as e:
            self._handle_error(e, "continue_after_tools")

    async def run(self) -> None:
        """Main chat loop."""
        # Initialize MCP connections
        await self.initialize()

        self.console.print()
        self.console.print(
            f"[bold]Groq CLI[/bold] - Model: {self.model_name}",
            style=INFO_STYLE,
        )
        self.console.print("[dim]Type /help for commands, Ctrl+D to exit[/dim]")
        self.console.print()

        try:
            while self.running:
                try:
                    # Get user input (sync - runs in thread)
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: Prompt.ask("[bold blue]You[/bold blue]")
                    )
                    if not user_input.strip():
                        continue

                    # Check for commands
                    if user_input.startswith("/"):
                        if self._handle_command(user_input):
                            continue

                    # Check for bare number (shortcut for last listed items)
                    if user_input.strip().isdigit():
                        idx = int(user_input.strip()) - 1

                        # Check if it's for models
                        if hasattr(self, '_last_model_list') and self._last_model_list:
                            if 0 <= idx < len(self._last_model_list):
                                selected = self._last_model_list[idx]
                                self._set_model(selected.name)
                                self._last_model_list = None  # Clear after use
                                continue
                            else:
                                self.console.print(f"[error]Invalid number. Choose 1-{len(self._last_model_list)}[/error]", style=ERROR_STYLE)
                                continue

                        # Check if it's for MCP servers (multi-digit like "134")
                        if hasattr(self, '_last_mcp_list') and self._last_mcp_list:
                            self._list_mcp(user_input.strip())  # Delegate to handler
                            continue

                    # Regular chat message
                    self.console.print()
                    await self._send_message(user_input)
                    self.console.print()

                except EOFError:
                    self.console.print("\n[dim]Goodbye![/dim]")
                    break
                except KeyboardInterrupt:
                    self.console.print()
                    continue
        finally:
            # Cleanup
            await self.shutdown()


async def async_main(model: str | None, mcp_config: Path | None) -> None:
    """Async entry point."""
    chat = GroqChat(model=model, mcp_config=mcp_config)
    await chat.run()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Groq CLI - Terminal chat with MCP tool support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  groq-chat                              Start with default model
  groq-chat --model llama-3.1-8b-instant Use specific model
  groq-chat --mcp-config ./servers.json  Custom MCP config
""",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=f"Groq model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--mcp-config",
        "-c",
        type=Path,
        default=None,
        help="Path to MCP servers config JSON",
    )

    args = parser.parse_args()

    # Handle signals
    def signal_handler(sig, frame):
        print("\nExiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run async
    try:
        asyncio.run(async_main(args.model, args.mcp_config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
