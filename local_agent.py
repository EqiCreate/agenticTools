from typing import Any, cast
from pathlib import Path
import sys
import httpx

from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


BASE_DIR = Path(__file__).parent.resolve()

http_client = httpx.AsyncClient(
    timeout=120.0,
    trust_env=False,
)

model = OllamaModel(
    "qwen3:4b-instruct", #qwen3:4b think-only
    provider=OllamaProvider(
        base_url="http://127.0.0.1:11434/v1",
        http_client=cast(Any, http_client),
    ),
)

transport = StdioTransport(
    command=sys.executable,
    args=[str(BASE_DIR / "mcp_server.py")],
)

toolset = MCPToolset(transport)

agent = Agent(
    model,
    toolsets=[toolset],
    instructions="""
You are a research assistant.

Use MCP tools when the question depends on experiment data
or exact arithmetic.
""",
)

result = agent.run_sync(
    """
Get the three FKG experiment scores.
Calculate their average using the calculate tool.
Tell me which experiment is highest.
"""
)

print(result.output)
print(result.usage)