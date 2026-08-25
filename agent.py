"""Tool-using agent: tools, schemas, and the control loop."""
import json
import re

import llm
from config import RELEVANCE_THRESHOLD, AGENT_MAX_ITERS
from retriever import retrieve as _retrieve
from ingest import list_documents as _list_documents


def retrieve_tool(query, source_filter=None):
    """Two-stage retrieval over the uploaded documents."""
    result = _retrieve(query, source_filter=source_filter)
    chunks = result["chunks"]
    top_score = chunks[0]["score"] if chunks else None
    weak = top_score is None or top_score < RELEVANCE_THRESHOLD
    return {
        "results": [
            {"text": c["text"], "source": c["source"], "page": c["page_number"], "score": c["score"]}
            for c in chunks
        ],
        "top_score": top_score,
        "hint": (
            "Low relevance — if these passages don't answer the question, reformulate the "
            "query and call retrieve again."
            if weak else
            "Relevant passages found."
        ),
    }


def list_documents_tool():
    """List the filenames currently available to answer from."""
    return {"documents": _list_documents()}


TOOLS = {"retrieve": retrieve_tool, "list_documents": list_documents_tool}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": (
                "Search the uploaded documents for passages relevant to a query. "
                "Call multiple times with different queries for multi-part questions. "
                "If results are low-relevance, reformulate and call again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A focused search query."},
                    "source_filter": {"type": "string", "description": "Optional filename to restrict the search."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the filenames of all uploaded documents available to answer from.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Provide the final answer. Every claim must be supported by retrieved passages. "
                "If the documents do not contain the answer, set answer to an explicit refusal and "
                "citations to an empty list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The grounded answer, or an explicit refusal."},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "page": {"type": "integer"},
                            },
                            "required": ["source", "page"],
                        },
                        "description": "Source+page for each passage used. Empty if refusing.",
                    },
                },
                "required": ["answer", "citations"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a document-grounded research assistant.

You answer ONLY from the uploaded documents. Use the `retrieve` tool to find relevant
passages; for multi-part questions, call it several times with different queries. If a
retrieval is low-relevance, reformulate the query and retrieve again before giving up.

When you can answer, call `finish` with the answer and a citation (source + page) for
every passage you used. If, after searching, the documents do not contain the answer,
call `finish` with an explicit refusal and an empty citations list. Never use outside
knowledge. Never invent citations."""

_REFUSAL = "I couldn't find enough relevant information in your documents to answer that."


def _assistant_message(resp):
    msg = {"role": "assistant", "content": resp["content"]}
    if resp["tool_calls"]:
        msg["tool_calls"] = [
            {"id": t["id"], "type": "function",
             "function": {"name": t["name"], "arguments": t["arguments"]}}
            for t in resp["tool_calls"]
        ]
    return msg


def _citations_from_trace(trace, limit=4):
    """Citations are the pages the retriever actually surfaced with strong relevance
    — accurate by construction, unlike the model's self-reported citations."""
    seen, cites = set(), []
    for step in trace:
        if step["tool"] != "retrieve":
            continue
        res = step["result"]
        top = res.get("top_score")
        if top is None or top < RELEVANCE_THRESHOLD:
            continue  # weak retrieval — not a real source
        for r in res.get("results", [])[:1]:  # the top hit of each strong retrieval
            key = (r["source"], r["page"])
            if key not in seen:
                seen.add(key)
                cites.append({"source": r["source"], "page": r["page"]})
    return cites[:limit]


def _extract_answer(content):
    """Some free models emit the finish payload as a JSON string in the message
    content instead of a tool call; pull the answer text out when that happens."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "answer" in data:
            return data["answer"]
    except (json.JSONDecodeError, TypeError):
        pass
    return content


# A citation array the model sometimes appends inline at the end of its answer text,
# e.g. `... why an email was flagged [{"source": "...", "page": 5}].`
_TRAILING_CITES = re.compile(
    r'\s*[\[【]\s*\{[^}]*"(?:source|page)"[^}]*\}.*?[\]】]\s*[.。]?\s*\Z', re.DOTALL
)


def _clean_answer(text):
    """Strip the inline citation JSON and a leading '**Answer:**' label the model
    sometimes adds — citations are shown separately as chips."""
    text = (text or "").strip()
    text = _TRAILING_CITES.sub("", text).strip()
    text = re.sub(r"^\*\*\s*answer\s*:?\s*\*\*\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def _finalize(answer, trace):
    """Attach trace-derived citations; refuse if nothing relevant was retrieved."""
    citations = _citations_from_trace(trace)
    if not citations:
        return {"answer": _REFUSAL, "citations": [], "trace": trace, "refused": True}
    return {"answer": _clean_answer(answer) or _REFUSAL, "citations": citations,
            "trace": trace, "refused": False}


def run(question, source_filter=None, history=None, chat=llm.chat, max_iters=AGENT_MAX_ITERS):
    """Drive the agent loop. Returns {answer, citations, trace, refused}."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history[-6:]
    messages.append({"role": "user", "content": question})

    trace = []
    tool_cache = {}  # (tool, args) -> result, so repeated searches are free
    for i in range(max_iters):
        # On the final allowed step, force the agent to answer from what it has
        # gathered rather than retrieving again and running out of steps.
        last_step = i == max_iters - 1
        if last_step:
            messages.append({
                "role": "system",
                "content": (
                    "This is your final step. Based on the passages already retrieved, "
                    "call finish now with your best grounded answer and citations. "
                    "Do not call retrieve again."
                ),
            })
            tool_choice = {"type": "function", "function": {"name": "finish"}}
        else:
            tool_choice = None

        resp = chat(messages, tools=TOOL_SCHEMAS, tool_choice=tool_choice)

        if not resp["tool_calls"]:
            # Model answered as plain text instead of calling finish — accept it.
            return _finalize(_extract_answer(resp["content"] or ""), trace)

        messages.append(_assistant_message(resp))

        for call in resp["tool_calls"]:
            name = call["name"]
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "finish":
                return _finalize(args.get("answer"), trace)

            func = TOOLS.get(name)
            # Free-tier round-trips are slow, so never pay for the same search
            # twice: serve a repeated query from cache and nudge the model on.
            cache_key = (name, json.dumps(args, sort_keys=True))
            if cache_key in tool_cache:
                result = dict(tool_cache[cache_key])
                result["hint"] = ("You already ran this exact query. Use the passages "
                                  "you have, or try a clearly different query.")
            else:
                try:
                    result = func(**args) if func else {"error": f"unknown tool: {name}"}
                except Exception as e:  # malformed args from the model shouldn't crash the request
                    result = {"error": f"tool '{name}' failed: {e}"}
                tool_cache[cache_key] = result
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "name": name, "content": json.dumps(result)})

    # Iteration cap reached → answer from what was gathered, or refuse if nothing relevant.
    return _finalize(None, trace)
