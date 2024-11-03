
# Mistral RAG Chatbot

## Overview
The Mistral RAG Chatbot is an AI-powered application designed for document indexing and querying. It leverages FAISS (Facebook AI Similarity Search) for efficient indexing and retrieval of documents, enabling intelligent search and answer capabilities for documents in PDF and DOC formats. The project integrates language models like Mistral and LLaMA for enhanced response generation, using the top search results as context.

## Features
- Indexing and searching PDF and DOC files using FAISS for fast, scalable vector search.
- Integration with Mistral and LLaMA language models for contextual response generation.
- Persistent storage of FAISS indexes for reloading between sessions.
- REST API endpoints for updating and querying indexes.
- Containerized setup using Docker and Docker Compose.
- GPU support for efficient model execution.

## Project Structure
```
ProjectPy/
├── .env.example
├── .gitignore
├── docker-compose.yaml
├── llama-7b-hf               # Store - Llama model
├── mistral-7b-instruct       # Store - Mistral model
├── Mistral RAG Chatbot.postman_collection.json
├── workspace/
│   ├── app.py                # Main Flask application
│   ├── base_model_handler.py # Handles model loading and management
│   ├── doc_indexer.py        # Module for indexing DOC files with FAISS
│   ├── legacy.py             # Legacy code support
│   ├── llama_module.py       # Module for LLaMA model handling
│   ├── mistral_module.py     # Module for Mistral model handling
│   ├── model_utils.py        # Utility functions for model operations
│   ├── pdf_indexer.py        # Module for indexing PDF files with FAISS
│   ├── searcher.py           # Handles FAISS-based search functionality
│   ├── shared_state.py       # Maintains shared state across processes
│   └── feed/
│       └── pdf/              # Directory for storing PDF documents
│           ├── Summary of 12 PCI-DSS Requirements - Deeper Dive.pdf
│           └── Summary of 12 PCI-DSS Requirements.pdf
```

## Prerequisites
- Docker and Docker Compose
- Python 3.8 or later (for local development)
- NVIDIA GPU (for GPU-enabled container execution)

## Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/miloskec/rag-faiss
   cd ProjectPy
   ```

2. **Set up the environment**:
   - Copy the `.env.example` file and create a `.env` file:
     ```bash
     cp .env.example .env
     ```

3. **Run the Docker container**:
   ```bash
   docker-compose up --build
   ```

4. **Access the API**:
   - The API will be available at `http://localhost:5000`.

## API Endpoints
- **`GET /update-index`**: Updates the FAISS index for PDF or DOC files.
  - Parameters: `type` (optional [`doc`, `docx`, `pdf`], defaults to `pdf`) 
  - **Details**: Indexed data is saved and can be reloaded for future use.

- **`POST /search`**: Searches the FAISS index and returns the top N results to the language model for contextual responses.
  - Request body:
    ```json
    {
      "query": "Your search query here"
    }
    ```
  - **Details**: The top N results from the FAISS search are used as context for generating model responses.

## Usage
- Place PDF files in `workspace/feed/pdf/` and DOC files in `workspace/feed/doc/`.
- Use the `/update-index` endpoint to index new documents.
- Search indexed documents using the `/search` endpoint to receive responses with relevant context from top search results.

## Contributing
Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License
This project is licensed under [MIT License](https://choosealicense.com/licenses/mit/).

## Contact
For further assistance, please contact [Milos Kecman].