from pathlib import Path

from pydantic_ai import Agent


BASE_DIR = Path(__file__).parent.resolve()


agent = Agent(
    "openai-codex:gpt-5.6-luna",
    instructions="""
You are a research assistant with several tools.

Rules:
1. If the answer depends on file content, use a file tool.
2. If exact arithmetic is needed, use calculate.
3. Do not invent file contents.
4. You may call multiple tools when necessary.
5. If no tool is needed, answer directly.
""",
)


def safe_path(filename: str) -> Path:
    path = (BASE_DIR / filename).resolve()

    if BASE_DIR not in path.parents and path != BASE_DIR:
        raise ValueError("Access outside the workspace is not allowed.")

    return path


@agent.tool_plain
def calculate(expression: str) -> str:
    """Calculate a simple arithmetic expression."""
    print(f"\n[TOOL] calculate({expression!r})")

    allowed = set("0123456789+-*/(). %")

    if not set(expression) <= allowed:
        return "Unsupported expression."

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


@agent.tool_plain
def read_file(filename: str) -> str:
    """Read the complete contents of a text file in the workspace."""
    print(f"\n[TOOL] read_file({filename!r})")

    path = safe_path(filename)

    if not path.exists():
        return f"File does not exist: {filename}"

    return path.read_text(encoding="utf-8")


@agent.tool_plain
def search_text(filename: str, query: str) -> str:
    """Search for lines containing a query inside a workspace text file."""
    print(f"\n[TOOL] search_text(filename={filename!r}, query={query!r})")

    path = safe_path(filename)

    if not path.exists():
        return f"File does not exist: {filename}"

    lines = path.read_text(encoding="utf-8").splitlines()

    matches = [
        f"{i}: {line}"
        for i, line in enumerate(lines, start=1)
        if query.lower() in line.lower()
    ]

    if not matches:
        return "No matches."

    return "\n".join(matches)


def run_test(question: str):
    print("\n" + "=" * 70)
    print("USER:", question)

    result = agent.run_sync(question)

    print("\nFINAL ANSWER:")
    print(result.output)

    print("\nMESSAGE TRACE:")
    for message in result.new_messages():
        print(message)

    print("\nUSAGE:")
    usage = result.usage

    print("requests:", usage.requests)
    print("tool_calls:", usage.tool_calls)
    print("input_tokens:", usage.input_tokens)
    print("output_tokens:", usage.output_tokens)
    print("total_tokens:", usage.total_tokens)
    print("cost:", usage.cost)


if __name__ == "__main__":
    run_test(
        """
Read notes.txt.

Find the three FKG experiment scores.
Calculate their average.
Also tell me which experiment has the highest score.

You must obtain the numbers from the file
and use the calculator for the arithmetic.
"""
    )
