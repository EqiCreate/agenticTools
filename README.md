# agenticTools

A small Python playground for learning how AI agents call local tools and MCP
servers. It includes examples using PydanticAI, OpenAI-compatible APIs, Ollama,
and a lightweight agent evaluation suite.

## What is included

- `agent.py`: PydanticAI agent with in-process file and calculator tools.
- `mcp_server.py`: local stdio MCP server exposing file, calculator, and FKG
  experiment tools.
- `mcp_agent.py`: cloud-model agent connected to the MCP server.
- `local_agent.py`: Ollama agent connected to the MCP server.
- `eval_agents.py`: agent benchmark and a focused trace for
  `Qwen3-4B-Instruct / T3_two_step`.
- `*_basic.py`: minimal HTTPX, OpenAI, and Ollama examples.
- `agent_eval_*.csv`: saved benchmark results.

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For local examples, install Ollama and make sure the configured model is
available:

```bash
ollama pull qwen3:4b
ollama serve
```

Cloud-model examples require the credentials expected by their configured
PydanticAI provider.

## Run

Run the MCP-backed local example:

```bash
python local_agent.py
```

Run the current focused agent evaluation:

```bash
python eval_agents.py
```

The current evaluation entry point runs `T3_two_step` once and prints each
model response's usage, tool calls, tool returns, text preview, and elapsed
time. The broader benchmark remains available in `main_old()` for reference.

## Notes

The calculator accepts only a restricted set of arithmetic characters and
evaluates expressions without built-ins. File tools resolve paths within the
project directory.
