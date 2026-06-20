"""Thin OpenRouter chat client with native tool-calling, normalized to plain dicts."""
from config import OPENROUTER_API_KEY, AGENT_MODEL

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client = None


def _get_client():
    global _client
    if _client is None:
        # Imported lazily so importing this module (at server startup) doesn't
        # pull in the heavy openai SDK — it's only needed when a question is asked.
        from openai import OpenAI
        _client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


def chat(messages, tools=None, model=None):
    """Call the model once. Returns {'content': str|None, 'tool_calls': [...]}.

    Each tool_call is {'id', 'name', 'arguments'} where arguments is a raw JSON string.
    """
    kwargs = {"model": model or AGENT_MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    message = _get_client().chat.completions.create(**kwargs).choices[0].message
    tool_calls = [
        {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
        for tc in (message.tool_calls or [])
    ]
    return {"content": message.content, "tool_calls": tool_calls}
