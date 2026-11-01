from typing import Any, cast

import httpx

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider


http_client = httpx.AsyncClient(
    timeout=60.0,
    trust_env=False,
)

model = OllamaModel(
    "qwen3:4b",
    provider=OllamaProvider(
        base_url="http://127.0.0.1:11434/v1",
        http_client=cast(Any, http_client),
    ),
)

agent = Agent(model)

result = agent.run_sync(
    "只回复一句：Ollama + PydanticAI works."
)

print(result.output)
print(result.usage)