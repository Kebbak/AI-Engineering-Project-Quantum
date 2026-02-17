from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import chromadb
from rag_qa import answer_query, DEFAULT_TOP_K, DEFAULT_MAX_SENTENCES

CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS)

# New endpoint to provide a corpus-based greeting
@app.route("/init", methods=["GET"])
def init_greeting():
    # Connect to ChromaDB and fetch a random or first document as greeting
    client = chromadb.PersistentClient(path="chroma_db")
    try:
        collection = client.get_collection(name="policy_chunks")
        # Fetch the first document (or random if you prefer)
        docs = collection.get(limit=1)
        if docs and docs.get("documents") and docs["documents"][0]:
            greeting = docs["documents"][0][0]
        else:
            greeting = "Welcome! I can answer questions about company policies."
    except Exception:
        greeting = "Welcome! I can answer questions about company policies."
    return jsonify({"greeting": greeting})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data.get("question")
        from rag_qa import answer_query, DEFAULT_TOP_K, DEFAULT_MAX_SENTENCES
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        answer_query(question, DEFAULT_TOP_K, False, None, DEFAULT_MAX_SENTENCES)
        sys.stdout = old_stdout
        output = mystdout.getvalue()
        # Debug log
        print("[DEBUG] answer_query output:\n", output)
        answer = output.split("Answer:")[-1].split("Sources:")[0].strip() if "Answer:" in output else output.strip()
        citations = []
        snippets = []
        if "Sources:" in output:
            sources_section = output.split("Sources:")[-1].strip()
            # Only include citations that match known corpus file patterns
            known_patterns = ["policy", ".md", ".txt", ".pdf", ".html", "code_of_conduct"]
            citations = [
                line.strip()
                for line in sources_section.split("\n")
                if line.strip()
                and not line.startswith("(none")
                and any(pat in line.strip() for pat in known_patterns)
            ]
        return jsonify({
            "answer": answer,
            "citations": citations,
            "snippets": snippets
        })
    except Exception as e:
        print("[ERROR] /chat endpoint exception:", e)
        return jsonify({"answer": "Sorry, something went wrong.", "citations": [], "snippets": []}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)