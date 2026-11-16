import logging
import os
from datetime import datetime

from flask import Flask, jsonify, request

import doc_indexer
import pdf_indexer
import searcher
from index_utils import IndexCompatibilityError
from rag_config import (
    DEFAULT_RAG_MODE,
    DEFAULT_TOP_K,
    FEED_DIR,
    LOG_DIR,
    SYNTHETIC_CORPUS_DIR,
    env_flag,
    normalize_rag_mode,
)

def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/update-index")
    def update_index():
        index_type = (request.args.get("type") or "pdf").strip().lower()
        try:
            if index_type in {"doc", "docx", "odt"}:
                count = doc_indexer.rebuild_doc_index(FEED_DIR / "doc")
            elif index_type == "synthetic":
                count = pdf_indexer.rebuild_text_index(SYNTHETIC_CORPUS_DIR)
            elif index_type == "pdf":
                count = pdf_indexer.rebuild_pdf_index(FEED_DIR / "pdf")
            else:
                return jsonify({"error": "type must be pdf, doc, or synthetic"}), 400
        except IndexCompatibilityError as error:
            logging.exception("Index rebuild failed")
            return jsonify({"error": str(error)}), 409
        except Exception as error:
            logging.exception("Index rebuild failed")
            return jsonify({"error": f"Index rebuild failed: {error}"}), 500

        return jsonify(
            {"status": "updated", "type": index_type, "chunks_indexed": count}
        )

    @app.post("/search")
    def search():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "JSON object body is required"}), 400

        query = data.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return jsonify({"error": "No query provided"}), 400
        if len(query) > 4000:
            return jsonify({"error": "query is too long"}), 400

        context_type = data.get("context_type", "pdf")
        if not isinstance(context_type, str) or context_type not in {
            "pdf",
            "synthetic",
            "doc",
            "docx",
            "odt",
        }:
            return jsonify({"error": "Invalid context_type"}), 400
        mode_value = data.get("mode", DEFAULT_RAG_MODE)
        if not isinstance(mode_value, str):
            return jsonify({"error": "mode must be a string"}), 400
        try:
            mode = normalize_rag_mode(mode_value)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        top_k = data.get("top_k", DEFAULT_TOP_K)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            return jsonify({"error": "top_k must be an integer between 1 and 100"}), 400
        model = data.get("model")
        if model is not None and not isinstance(model, str):
            return jsonify({"error": "model must be a string"}), 400
        template = data.get("template")
        if template is not None and not isinstance(template, str):
            return jsonify({"error": "template must be a string"}), 400

        try:
            result = searcher.search_index(
                model,
                query.strip(),
                template=template,
                context_type=context_type,
                top_k=top_k,
                mode=mode,
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        except IndexCompatibilityError as error:
            return jsonify({"error": str(error)}), 409
        except Exception as error:
            logging.exception("Search failed")
            return jsonify({"error": f"Search failed: {error}"}), 500

        return jsonify({"query": query.strip(), **result})

    @app.post("/clean-resources")
    def unload_current_model():
        try:
            result = searcher.unload_model()
        except Exception as error:
            logging.exception("Model unload failed")
            return jsonify({"error": str(error)}), 500
        return jsonify(result)

    return app


app = create_app()


def configure_logging():
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(
            filename=LOG_DIR / f"main_{timestamp}.log",
            level=logging.INFO,
            format=log_format,
            force=True,
        )
    except OSError as error:
        logging.basicConfig(level=logging.INFO, format=log_format, force=True)
        logging.warning("File logging unavailable (%s); logging to stderr", error)


def main():
    pdf_indexer.initialize()
    doc_indexer.initialize()
    configure_logging()
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=env_flag("FLASK_DEBUG", default=False),
    )


if __name__ == "__main__":
    main()
