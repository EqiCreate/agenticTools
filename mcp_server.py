from pathlib import Path

from mcp.server import MCPServer


BASE_DIR = Path(__file__).parent.resolve()

mcp = MCPServer("AI Playground Tools")


def safe_path(filename: str) -> Path:
    path = (BASE_DIR / filename).resolve()

    if BASE_DIR not in path.parents and path != BASE_DIR:
        raise ValueError("Access outside workspace is not allowed.")

    return path


@mcp.tool()
def calculate(expression: str) -> str:
    """Calculate a simple arithmetic expression."""

    allowed = set("0123456789+-*/(). %")

    if not set(expression) <= allowed:
        return "Unsupported expression."

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Calculation error: {e}"


@mcp.tool()
def read_file(filename: str) -> str:
    """Read a text file from the workspace."""

    path = safe_path(filename)

    if not path.exists():
        return f"File does not exist: {filename}"

    return path.read_text(encoding="utf-8")


@mcp.tool()
def search_text(filename: str, query: str) -> str:
    """Search for lines containing text in a workspace file."""

    path = safe_path(filename)

    if not path.exists():
        return f"File does not exist: {filename}"

    lines = path.read_text(encoding="utf-8").splitlines()

    matches = [
        f"{i}: {line}"
        for i, line in enumerate(lines, start=1)
        if query.lower() in line.lower()
    ]

    return "\n".join(matches) if matches else "No matches."

@mcp.tool()
def get_fkg_experiments() -> dict:
    """Return the FKG experiment scores from the current experiment dataset."""

    return {
        "experiments": [
            {"name": "experiment_a", "score": 0.82},
            {"name": "experiment_b", "score": 0.76},
            {"name": "experiment_c", "score": 0.91},
        ],
        "metric": "score",
    }

@mcp.tool()
def get_best_fkg_experiment() -> dict:
    """Return the highest-scoring FKG experiment."""

    experiments = [
        {"name": "experiment_a", "score": 0.82},
        {"name": "experiment_b", "score": 0.76},
        {"name": "experiment_c", "score": 0.91},
    ]

    best = max(experiments, key=lambda x: x["score"])

    return best

if __name__ == "__main__":
    mcp.run()