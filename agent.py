"""Tool-using agent: tools, schemas, and the control loop."""
import json

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


def run(question, source_filter=None, history=None, chat=llm.chat, max_iters=AGENT_MAX_ITERS):
    """Drive the agent loop. Returns {answer, citations, trace, refused}."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history[-6:]
    messages.append({"role": "user", "content": question})

    trace = []
    for _ in range(max_iters):
        resp = chat(messages, tools=TOOL_SCHEMAS)

        if not resp["tool_calls"]:
            # Model answered without calling finish — accept its content as the answer.
            return {"answer": resp["content"] or _REFUSAL, "citations": [],
                    "trace": trace, "refused": not resp["content"]}

        messages.append(_assistant_message(resp))

        for call in resp["tool_calls"]:
            name = call["name"]
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "finish":
                citations = args.get("citations", [])
                return {"answer": args.get("answer", _REFUSAL), "citations": citations,
                        "trace": trace, "refused": len(citations) == 0}

            func = TOOLS.get(name)
            result = func(**args) if func else {"error": f"unknown tool: {name}"}
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "name": name, "content": json.dumps(result)})

    # Iteration cap reached without finishing → grounded refusal.
    return {"answer": _REFUSAL, "citations": [], "trace": trace, "refused": True}
