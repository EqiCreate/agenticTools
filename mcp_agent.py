import sys
from pathlib import Path

from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset


BASE_DIR = Path(__file__).parent.resolve()
SERVER_PATH = BASE_DIR / "mcp_server.py"


transport = StdioTransport(
    command=sys.executable,
    args=[str(SERVER_PATH)],
)

toolset = MCPToolset(transport)


agent = Agent(
    "openai-codex:gpt-5.6-luna",
    toolsets=[toolset],
    instructions="""
You are a research assistant.

Use MCP tools when the question depends on workspace files
or exact arithmetic.
""",
)


if __name__ == "__main__":
    result = agent.run_sync(
        """
Read notes.txt.

Find the three FKG experiment scores.
Calculate their average.

Use the available tools.
"""
    )

    print(result.output)
    print(result.usage)