import httpx

response = httpx.post(
    "http://127.0.0.1:11434/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer ollama",
    },
    json={
        "model": "qwen3:4b",
        "messages": [
            {"role": "user", "content": "只回复 OK"}
        ],
        "reasoning_effort": "none",
    },
    timeout=60.0,
    trust_env=False,
)

print(response.status_code)
print(response.text)