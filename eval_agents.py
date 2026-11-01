from __future__ import annotations
import time
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import httpx

from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


BASE_DIR = Path(__file__).parent.resolve()
SERVER_PATH = BASE_DIR / "mcp_server.py"
REPEATS = 3

# ---------------------------------------------------------
# 1. Eval cases
# ---------------------------------------------------------

@dataclass
class EvalCase:
    name: str
    prompt: str

    # What tool calls would be ideal?
    expected_tools: list[str]

    # Simple deterministic check of final answer.
    answer_check: Callable[[str], bool]


CASES = [
    EvalCase(
        name="T1_no_tool",
        prompt="""
简单解释什么是联邦学习。
不要访问 experiment 数据。
""",
        expected_tools=[],
        answer_check=lambda x: "联邦" in x,
    ),

    EvalCase(
        name="T2_single_tool",
        prompt="""
告诉我 experiment_b 的 score。
请从 FKG experiment 数据中获取。
""",
        expected_tools=["get_fkg_experiments"],
        answer_check=lambda x: (
            "experiment_b" in x.lower()
            and "0.76" in x
        ),
    ),

    EvalCase(
        name="T3_two_step",
        prompt="""
获取所有 FKG experiment score。
使用 calculate 工具计算最高分减最低分。
""",
        expected_tools=[
            "get_fkg_experiments",
            "calculate",
        ],
        answer_check=lambda x: (
            "0.15" in x
            or "0.150" in x
        ),
    ),

    EvalCase(
        name="T4_specialized_tool",
        prompt="""
告诉我当前最高分的 FKG experiment。
""",
        expected_tools=[
            "get_best_fkg_experiment",
        ],
        answer_check=lambda x: (
            "experiment_c" in x.lower()
            and "0.91" in x
        ),
    ),

    EvalCase(
        name="T5_composed_task",
        prompt="""
获取所有 FKG experiment。

告诉我最高分 experiment，
并使用 calculate 计算三个 score 的平均值。
""",
        expected_tools=[
            "get_fkg_experiments",
            "calculate",
        ],
        answer_check=lambda x: (
            "experiment_c" in x.lower()
            and "0.91" in x
            and "0.83" in x
        ),
    ),
]


# ---------------------------------------------------------
# 2. Shared MCP
# ---------------------------------------------------------

def make_toolset() -> MCPToolset:
    transport = StdioTransport(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )
    return MCPToolset(transport)


# ---------------------------------------------------------
# 3. Models
# ---------------------------------------------------------

def make_cloud_agent() -> Agent:
    return Agent(
        "openai-codex:gpt-5.6-luna",
        toolsets=[make_toolset()],
        instructions="""
You are an experiment analysis agent.

Use MCP tools only when the task requires external experiment
data or exact calculation.

Avoid unnecessary tool calls.
""",
    )


def make_local_agent() -> Agent:
    # Temporary workaround for the Ollama 0.34.4 / HTTPX2
    # compatibility issue we diagnosed.
    http_client = httpx.AsyncClient(
        timeout=120.0,
        trust_env=False,
    )

    model = OllamaModel(
        "qwen3:4b",
        provider=OllamaProvider(
            base_url="http://127.0.0.1:11434/v1",
            http_client=cast(Any, http_client),
        ),
    )

    return Agent(
        model,
        toolsets=[make_toolset()],
        instructions="""
You are an experiment analysis agent.

Use MCP tools only when the task requires external experiment
data or exact calculation.

Avoid unnecessary tool calls.
""",
    )


# ---------------------------------------------------------
# 4. Inspect tool calls
# ---------------------------------------------------------

def extract_tool_calls(result) -> list[str]:
    tool_names: list[str] = []

    # PydanticAI exposes the complete model/tool message history.
    for message in result.all_messages():
        if not isinstance(message, ModelResponse):
            continue

        for part in message.parts:
            if isinstance(part, ToolCallPart):
                tool_names.append(part.tool_name)

    return tool_names


def _trace_preview(value: Any, limit: int = 500) -> str:
    text = str(value)
    return text[:limit]


def print_t3_trace(result) -> list[dict[str, Any]]:
    """Print a minimal per-request trace for T3_two_step only."""
    trace: list[dict[str, Any]] = []
    request_sent_at = None

    print()
    print("T3_two_step TRACE")
    print("-" * 75)

    for message in result.all_messages():
        if isinstance(message, ModelRequest):
            request_sent_at = message.timestamp

            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    print("ToolReturnPart:")
                    print(f"  tool_name: {part.tool_name}")
                    print(
                        "  content[0:500]: "
                        f"{_trace_preview(part.content)}"
                    )

        elif isinstance(message, ModelResponse):
            step = len(trace) + 1
            duration_seconds = None
            if request_sent_at is not None:
                duration_seconds = (
                    message.timestamp - request_sent_at
                ).total_seconds()

            tool_names: list[str] = []

            print()
            print(f"ModelResponse request/step: {step}")
            print(f"  usage: {message.usage}")
            if duration_seconds is not None:
                print(
                    "  model_elapsed_seconds: "
                    f"{duration_seconds:.2f}"
                )

            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    tool_names.append(part.tool_name)
                    print("  ToolCallPart:")
                    print(f"    tool_name: {part.tool_name}")
                    print(f"    args: {part.args}")
                elif isinstance(part, TextPart):
                    print(
                        "  TextPart[0:500]: "
                        f"{_trace_preview(part.content)}"
                    )

            trace.append(
                {
                    "step": step,
                    "output_tokens": getattr(
                        message.usage,
                        "output_tokens",
                        None,
                    ),
                    "tools": tool_names,
                    "model_elapsed_seconds": duration_seconds,
                }
            )

    print()
    print("T3_two_step TRACE SUMMARY")
    print(f"  model_requests: {len(trace)}")
    for item in trace:
        elapsed = item["model_elapsed_seconds"]
        elapsed_text = (
            f"{elapsed:.2f}s"
            if elapsed is not None
            else "unknown"
        )
        print(
            f"  step {item['step']}: "
            f"output_tokens={item['output_tokens']}, "
            f"tools={item['tools']}, "
            f"model_elapsed={elapsed_text}"
        )

    if trace:
        timed_steps = [
            item
            for item in trace
            if item["model_elapsed_seconds"] is not None
        ]
        token_steps = [
            item
            for item in trace
            if item["output_tokens"] is not None
        ]
        if timed_steps:
            slowest = max(
                timed_steps,
                key=lambda item: item["model_elapsed_seconds"],
            )
            print(
                "  slowest_step_in_this_run: "
                f"{slowest['step']} "
                f"({slowest['model_elapsed_seconds']:.2f}s)"
            )
        if token_steps:
            largest = max(
                token_steps,
                key=lambda item: item["output_tokens"],
            )
            print(
                "  largest_output_step_in_this_run: "
                f"{largest['step']} "
                f"({largest['output_tokens']} tokens)"
            )

    return trace


# ---------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------

def evaluate_tools(
    actual: list[str],
    expected: list[str],
) -> tuple[bool, int]:
    """
    tool_correct:
        Every expected tool was used.

    extra_tool_calls:
        Number of calls beyond the expected calls.
        Repeated calls also count as extra.
    """

    remaining = actual.copy()

    for expected_tool in expected:
        if expected_tool not in remaining:
            return False, max(0, len(actual) - len(expected))

        remaining.remove(expected_tool)

    return True, len(remaining)


# ---------------------------------------------------------
# 6. Run one case
# ---------------------------------------------------------

def classify_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()

    if "timed out" in text or "timeout" in text:
        return "TIMEOUT"

    if "connection error" in text or "connecterror" in text:
        return "CONNECTION_ERROR"

    return "OTHER_ERROR"


def run_case(
    model_name: str,
    agent: Agent,
    case: EvalCase,
    repeat: int,
    total_repeats: int = REPEATS,
) -> dict[str, Any]:

    print()
    print("=" * 75)
    print(
        f"{model_name} | {case.name} "
        f"| repeat {repeat}/{total_repeats}"
    )
    print("=" * 75)

    start = time.perf_counter()

    try:
        result = agent.run_sync(case.prompt)

        latency = time.perf_counter() - start

        output = str(result.output)
        usage = result.usage

        tool_names = extract_tool_calls(result)

        if case.name == "T3_two_step":
            print_t3_trace(result)

        answer_correct = case.answer_check(output)

        tool_correct, extra_tools = evaluate_tools(
            tool_names,
            case.expected_tools,
        )

        success = answer_correct and tool_correct

        if success:
            status = "PASS"
        elif not answer_correct:
            status = "ANSWER_FAIL"
        else:
            status = "TOOL_FAIL"

        print("Output:")
        print(output)

        print()
        print("Tools:", tool_names)
        print("Expected:", case.expected_tools)
        print("Status:", status)
        print(f"Latency: {latency:.2f}s")

        return {
            "model": model_name,
            "case": case.name,
            "repeat": repeat,
            "status": status,
            "success": success,
            "answer_correct": answer_correct,
            "tool_correct": tool_correct,
            "tool_calls_actual": len(tool_names),
            "tool_calls_expected": len(case.expected_tools),
            "extra_tool_calls": extra_tools,
            "tools": " -> ".join(tool_names),
            "requests": getattr(usage, "requests", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_tokens": getattr(
                usage,
                "cache_read_tokens",
                None,
            ),
            "latency_seconds": round(latency, 2),
            "error": "",
        }

    except Exception as exc:
        latency = time.perf_counter() - start

        status = classify_error(exc)

        print("ERROR:", repr(exc))
        print("Status:", status)
        print(f"Latency: {latency:.2f}s")

        return {
            "model": model_name,
            "case": case.name,
            "repeat": repeat,
            "status": status,
            "success": False,
            "answer_correct": False,
            "tool_correct": False,
            "tool_calls_actual": 0,
            "tool_calls_expected": len(case.expected_tools),
            "extra_tool_calls": 0,
            "tools": "",
            "requests": None,
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_tokens": None,
            "latency_seconds": round(latency, 2),
            "error": repr(exc),
        }


# ---------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------

def print_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print()
    print("=" * 120)
    print("AGENT EVAL SUMMARY")
    print("=" * 120)

    header = (
        f"{'Model':<15}"
        f"{'Case':<22}"
        f"{'Run':<6}"
        f"{'Status':<20}"
        f"{'Tools':<8}"
        f"{'Extra':<8}"
        f"{'Req':<8}"
        f"{'Latency':<10}"
        f"{'Output':<10}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['model']:<15}"
            f"{row['case']:<22}"
            f"{row['repeat']:<6}"
            f"{row['status']:<20}"
            f"{row['tool_calls_actual']:<8}"
            f"{row['extra_tool_calls']:<8}"
            f"{str(row['requests']):<8}"
            f"{row['latency_seconds']:<10}"
            f"{str(row['output_tokens']):<10}"
        )

    print()
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    model_names = sorted(
        set(row["model"] for row in rows)
    )

    for model_name in model_names:
        model_rows = [
            row
            for row in rows
            if row["model"] == model_name
        ]

        total = len(model_rows)

        passes = sum(
            row["status"] == "PASS"
            for row in model_rows
        )

        timeouts = sum(
            row["status"] == "TIMEOUT"
            for row in model_rows
        )

        connection_errors = sum(
            row["status"] == "CONNECTION_ERROR"
            for row in model_rows
        )

        answer_failures = sum(
            row["status"] == "ANSWER_FAIL"
            for row in model_rows
        )

        tool_failures = sum(
            row["status"] == "TOOL_FAIL"
            for row in model_rows
        )

        other_errors = sum(
            row["status"] == "OTHER_ERROR"
            for row in model_rows
        )

        extra_tools = sum(
            row["extra_tool_calls"]
            for row in model_rows
        )

        latencies = [
            row["latency_seconds"]
            for row in model_rows
            if row["status"] == "PASS"
        ]

        avg_latency = (
            sum(latencies) / len(latencies)
            if latencies
            else 0
        )

        output_tokens = sum(
            row["output_tokens"] or 0
            for row in model_rows
        )

        print()
        print(model_name)
        print(f"  Runs:                 {total}")
        print(f"  PASS:                 {passes}/{total}")
        print(f"  Timeout:              {timeouts}")
        print(f"  Connection errors:    {connection_errors}")
        print(f"  Answer failures:      {answer_failures}")
        print(f"  Tool failures:        {tool_failures}")
        print(f"  Other errors:         {other_errors}")
        print(f"  Extra tool calls:     {extra_tools}")
        print(f"  Avg PASS latency:     {avg_latency:.2f}s")
        print(f"  Output tokens total:  {output_tokens}")
# ---------------------------------------------------------
# 8. Main
# ---------------------------------------------------------

def main_old():
    agents = {
        "GPT-5.6-Luna": make_cloud_agent(),
         "Qwen3-4B-Instruct": make_local_agent(),
        # "Qwen3-4B": make_local_agent(),
    }

    rows: list[dict[str, Any]] = []

    for model_name, agent in agents.items():
        for case in CASES:
            for repeat in range(1, REPEATS + 1):
                rows.append(
                    run_case(
                        model_name,
                        agent,
                        case,
                        repeat,
                    )
                )

    print_summary(rows)

    csv_path = BASE_DIR / "agent_eval_results_repeat3.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved results to: {csv_path}")

def main():
    agent = make_local_agent()

    case = next(
        case
        for case in CASES
        if case.name == "T3_two_step"
    )

    rows = [
        run_case(
            "Qwen3-4B-Instruct",
            agent,
            case,
            1,
            total_repeats=1,
        )
    ]

    print_summary(rows)

if __name__ == "__main__":
    main()
