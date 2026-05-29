import json
import requests
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are a document assistant. Answer the user's question using ONLY the context passages provided.

Rules:
- If the context contains the answer, provide it clearly and cite the page number(s) in brackets, e.g. [Page 5].
- If the context does not contain enough information, say exactly: "I don't have enough information in the uploaded documents to answer this question."
- Do not use any outside knowledge. Only use what is in the context.
- Be concise."""

MAX_HISTORY = 6  # keep last 3 exchanges (6 messages) to avoid token bloat


def _build_messages(question: str, chunks: list[dict], history: list[dict]) -> list[dict]:
    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += f"[Context {i+1} | {chunk['source']} | Page {chunk['page_number']}]\n"
        context_block += chunk["text"].strip() + "\n\n"

    user_message = f"Context:\n{context_block}\nQuestion: {question}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-MAX_HISTORY:]
    messages.append({"role": "user", "content": user_message})
    return messages


def generate_answer_stream(question: str, chunks: list[dict], history: list[dict] = None):
    """
    Yields SSE-formatted strings: token events then a final done event with sources.
    Usage: yield from generate_answer_stream(...)
    """
    if history is None:
        history = []

    if not chunks:
        msg = "I don't have enough information in the uploaded documents to answer this question."
        yield f"data: {json.dumps({'token': msg})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
        return

    messages = _build_messages(question, chunks, history)

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": OPENROUTER_MODEL, "messages": messages, "stream": True},
        stream=True,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        payload = decoded[6:]
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
            content = data["choices"][0]["delta"].get("content", "")
            if content:
                yield f"data: {json.dumps({'token': content})}\n\n"
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    seen = set()
    sources = []
    for c in chunks:
        key = (c["source"], c["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": c["source"], "page_number": c["page_number"]})

    yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
