from flask import Flask, request, jsonify
import pdf_indexer as pdf_indexer
import doc_indexer as doc_indexer
import searcher
import logging
from datetime import datetime
import os
# Initialize Flask app
app = Flask(__name__)

# Route to update the index
@app.route('/update-index', methods=['GET'])
def update_index():
    data = request.args
    type = data.get('type', 'pdf')
    print(f"Updating index for {type} files")
    if type == 'doc' or type == 'docx':
        doc_indexer.clear_index()
        docx_folder = "./workspace/feed/doc"
        doc_indexer.update_doc_index(docx_folder)
        doc_indexer.save_index_and_mappings()
    else:
        pdf_indexer.clear_index()
        pdf_folder = "./workspace/feed/pdf"
        pdf_indexer.update_pdf_index(pdf_folder)
        pdf_indexer.save_index_and_mappings()
        
    return jsonify({"status": "updated"})

# Route to search the index
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    template = data.get('template', '')
    context_type = data.get('context_type', 'pdf')
    model = data.get('model', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    search_results = searcher.search_index(model, query, template, context_type, top_k=5)
    return jsonify({"query": query, "results": search_results})

@app.route('/clean-resources', methods=['GET'])
# Unload current model
def unload_current_model():
    try:
        result = searcher.unload_model()
    except Exception as e:
        result = {"error": str(e)}
    return jsonify(result)
# Main entry point
if __name__ == "__main__":
    # Initialize FAISS index and mappings before running the app
    pdf_indexer.initialize()  # Load the index and mappings into shared state
    doc_indexer.initialize()  # Load the index and mappings into shared state

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"workspace/logs/main_{timestamp}.log"
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Configure logging to output to a file
    logging.basicConfig( 
        filename=f"workspace/logs/main_{timestamp}.log",  # The name of the log file
        level=logging.ERROR,  # The logging level
        format='%(asctime)s - %(levelname)s - %(message)s'  # The format of log messages
    )
    app.run(host='0.0.0.0', port=5000, debug=True)