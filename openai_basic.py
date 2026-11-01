from typing import Any, cast

import httpx
from openai import OpenAI


http_client = httpx.Client(
    timeout=60.0,
    trust_env=False,
)

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1/",
    api_key="ollama",
    max_retries=0,
    http_client=cast(Any, http_client),
)

response = client.chat.completions.create(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "只回复 OK",
        }
    ],
)

print(response.choices[0].message.content)