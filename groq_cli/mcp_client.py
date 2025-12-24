"""MCP (Model Context Protocol) client for tool execution.

Supports both stdio and HTTP transport for MCP servers.
Uses the official MCP SDK for proper async communication.
Adapted for OpenAI-compatible tool format.
"""

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

# Default MCP config location (shared with gemini-cli and shell-chat)
BACKEND_MCP_CONFIG = Path("/home/human/REPOS/Backend_FastAPI/data/mcp_servers.json")
HTTP_CONNECTION_TIMEOUT = 30.0


class MCPServer:
    """Manages a single MCP server connection using the official SDK."""

    def __init__(
        self,
        name: str,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        http_url: str | None = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd
        self.http_url = http_url
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[dict[str, Any]] = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def is_http(self) -> bool:
        return self.http_url is not None

    async def connect(self) -> bool:
        """Connect to the MCP server (HTTP or stdio)."""
        if self.http_url:
            return await self._connect_http()
        else:
            return await self._connect_stdio()

    async def _connect_http(self) -> bool:
        """Connect to an HTTP MCP server."""
        try:
            from mcp.client.streamable_http import streamablehttp_client
            import httpx

            logger.info(f"Connecting to HTTP MCP server: {self.name} at {self.http_url}")

            # First check if the server is reachable
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    await client.get(self.http_url)
            except Exception:
                logger.warning(f"HTTP MCP server {self.name} not reachable at {self.http_url}")
                return False

            self._exit_stack = AsyncExitStack()

            async with asyncio.timeout(HTTP_CONNECTION_TIMEOUT):
                http_manager = streamablehttp_client(self.http_url)
                read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                    http_manager
                )

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()

            # Fetch tools
            await self._refresh_tools()

            logger.info(f"Connected to HTTP MCP server: {self.name} ({len(self._tools)} tools)")
            return True

        except asyncio.TimeoutError:
            logger.warning(f"Timeout connecting to HTTP MCP server {self.name}")
            if self._exit_stack:
                try:
                    await self._exit_stack.aclose()
                except Exception:
                    pass
            self._exit_stack = None
            self._session = None
            return False
        except ImportError:
            logger.error("HTTP MCP transport not available - install mcp[streamable-http]")
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to HTTP MCP server {self.name}: {e}")
            if self._exit_stack:
                try:
                    await self._exit_stack.aclose()
                except Exception:
                    pass
            self._exit_stack = None
            self._session = None
            return False

    async def _connect_stdio(self) -> bool:
        """Connect to a stdio MCP server."""
        try:
            if not self.command:
                logger.error(f"No command specified for stdio server {self.name}")
                return False

            # Build environment
            full_env = os.environ.copy()
            full_env.update(self.env)

            # Create server parameters
            params = StdioServerParameters(
                command=self.command,
                args=self.args,
                cwd=self.cwd,
                env=full_env,
            )

            logger.info(f"Starting stdio MCP server: {self.name}")
            logger.debug(f"Command: {self.command} {' '.join(self.args)}")

            # Create exit stack for resource management
            self._exit_stack = AsyncExitStack()

            # Start the stdio client
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )

            # Create and initialize session
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()

            # Fetch tools
            await self._refresh_tools()

            logger.info(f"Connected to stdio MCP server: {self.name} ({len(self._tools)} tools)")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to stdio MCP server {self.name}: {e}")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning(f"Error closing MCP server {self.name}: {e}")
            finally:
                self._exit_stack = None
                self._session = None
                self._tools = []
                logger.info(f"Disconnected from MCP server: {self.name}")

    async def _refresh_tools(self) -> None:
        """Fetch available tools from the server."""
        if not self._session:
            return

        tools = []
        cursor = None

        while True:
            result = await self._session.list_tools(cursor=cursor)
            for tool in result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema or {"type": "object", "properties": {}},
                })
            cursor = result.nextCursor
            if not cursor:
                break

        self._tools = tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return the result as a string."""
        if not self._session:
            return f"Error: Not connected to MCP server {self.name}"

        try:
            result = await self._session.call_tool(tool_name, arguments)

            # Extract text from content blocks
            texts = []
            for item in result.content:
                data = item.model_dump()
                if item.type == "text":
                    value = data.get("text")
                    if isinstance(value, str):
                        texts.append(value)
                else:
                    texts.append(json.dumps(data))

            return "\n".join(texts) if texts else str(result)

        except Exception as e:
            logger.error(f"Tool call failed for {tool_name}: {e}")
            return f"Error: {e}"


class MCPClient:
    """Manages multiple MCP servers and provides unified tool access."""

    def __init__(self):
        self.servers: dict[str, MCPServer] = {}
        self._tool_to_server: dict[str, str] = {}

    def load_config(self, config_path: Path | None = None) -> bool:
        """Load MCP server configuration from JSON file.
        
        Supports both legacy format and Backend's mcp_servers.json format.
        """
        # Try to load from provided path or default Backend config
        if config_path is None:
            config_path = BACKEND_MCP_CONFIG

        if not config_path.exists():
            logger.warning(f"MCP config not found: {config_path}")
            return False

        try:
            with open(config_path) as f:
                config = json.load(f)

            # Handle Backend format: {"servers": [{...}, ...]}
            if "servers" in config and isinstance(config["servers"], list):
                return self._load_backend_format(config["servers"])
            
            # Handle legacy format: {"mcpServers": {...}} or {"servers": {...}}
            servers_config = config.get("servers", config.get("mcpServers", {}))
            if isinstance(servers_config, dict):
                return self._load_legacy_format(servers_config)

            logger.warning(f"Unknown MCP config format in {config_path}")
            return False

        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return False

    def _load_backend_format(self, servers: list[dict]) -> bool:
        """Load Backend's mcp_servers.json format."""
        count = 0
        for server_config in servers:
            server_id = server_config.get("id")
            if not server_id:
                continue

            # Only load enabled servers
            if not server_config.get("enabled", False):
                continue

            # Check if enabled for CLI (respects shared config with shell-chat/gemini-chat)
            client_enabled = server_config.get("client_enabled", {})
            if not client_enabled.get("cli", True):
                continue

            # Build HTTP URL from http_port if available
            http_port = server_config.get("http_port")
            http_url = server_config.get("http_url")
            
            if http_port and not http_url:
                http_url = f"http://127.0.0.1:{http_port}/mcp"

            if http_url:
                self.servers[server_id] = MCPServer(
                    name=server_id,
                    http_url=http_url,
                )
                count += 1
            elif server_config.get("module"):
                # For module-based servers, they should be run via start_mcp_servers.py
                logger.debug(f"Skipping module server {server_id} - use MCP pool")

        logger.info(f"Loaded {count} HTTP MCP servers from Backend config")
        return count > 0

    def _load_legacy_format(self, servers_config: dict) -> bool:
        """Load legacy config format."""
        for name, server_config in servers_config.items():
            command = server_config.get("command", "python")
            args = server_config.get("args", [])
            env = server_config.get("env", {})
            cwd = server_config.get("cwd")

            self.servers[name] = MCPServer(name, command, args, env, cwd)

        logger.info(f"Loaded {len(self.servers)} MCP servers from legacy config")
        return len(self.servers) > 0

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers."""
        for name, server in self.servers.items():
            if await server.connect():
                for tool in server.tools:
                    self._tool_to_server[tool["name"]] = name

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for server in self.servers.values():
            await server.disconnect()

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all tools from all servers in OpenAI function format."""
        tools = []
        for server in self.servers.values():
            for tool in server.tools:
                # OpenAI function calling format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                    }
                }
                tools.append(openai_tool)
        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool by name, routing to the appropriate server."""
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return f"Error: Unknown tool '{tool_name}'"

        server = self.servers.get(server_name)
        if not server:
            return f"Error: Server '{server_name}' not found"

        return await server.call_tool(tool_name, arguments)

    def list_servers(self) -> list[dict[str, Any]]:
        """List all servers and their status."""
        result = []
        for name, server in self.servers.items():
            result.append({
                "name": name,
                "running": server.is_connected,
                "tools": len(server.tools),
                "http": server.is_http,
            })
        return result
