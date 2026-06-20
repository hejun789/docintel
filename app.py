import os
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from ingest import extract_text_from_file, chunk_text, store_chunks, delete_document, list_documents
import agent

print("BOOT: app module loaded, Flask starting", flush=True)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/documents", methods=["GET"])
def get_documents():
    try:
        docs = list_documents()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"documents": docs})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        pages = extract_text_from_file(filepath)
        chunks = chunk_text(pages, file.filename)
        count = store_chunks(chunks)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(filepath)

    return jsonify({
        "message": f"'{file.filename}' ingested successfully.",
        "pages": len(pages),
        "chunks": count,
    })


@app.route("/document/<path:filename>", methods=["DELETE"])
def remove_document(filename):
    try:
        count = delete_document(filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"message": f"'{filename}' removed.", "chunks_deleted": count})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Request body must include a 'question' field"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    history = data.get("history", [])
    source_filter = data.get("document") or None

    result = agent.run(question, source_filter=source_filter, history=history)

    # The frontend renders sources as `source · p.{page_number}`; map the agent's
    # citation shape ({source, page}) onto that contract.
    sources = [{"source": c.get("source"), "page_number": c.get("page")} for c in result["citations"]]

    def stream():
        import json
        yield f"data: {json.dumps({'token': result['answer']})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'trace': result['trace']})}\n\n"

    return Response(stream_with_context(stream()), content_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
