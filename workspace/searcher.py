import numpy as np
from sentence_transformers import SentenceTransformer
import shared_state  # Import the shared index and mappings
from transformers import pipeline, AutoTokenizer
import torch
# Dynamic import based on the selected model
from model_utils import get_model_handler, unload_current_model # Function to dynamically select the model
from model_utils import query_model  # Function to query the selected model

# Check if a GPU is available
device = 0 if torch.cuda.is_available() else -1
# Initialize the reranking pipeline (using a pre-trained cross-encoder model)
reranker = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Initialize tokenizer

# Maximum sequence length supported by the model cross-encoder/ms-marco-MiniLM-L-6-v2
MAX_LENGTH = 512

def truncate_text(query, context, max_length=512):
    query_tokens = tokenizer.encode(query, add_special_tokens=False)
    context_tokens = tokenizer.encode(context, add_special_tokens=False)

    total_length = len(query_tokens) + len(context_tokens)

    if total_length > max_length:
        truncated_context_tokens = context_tokens[:max_length - len(query_tokens) - 8]  # -2 bc [CLS] i [SEP] also " [...]" = 6 tokens (-2-6)
        truncated_context = tokenizer.decode(truncated_context_tokens, skip_special_tokens=True) + " [...]"
        return truncated_context

    return context
# Initialize the sentence transformer model (to convert text to embeddings)
model = SentenceTransformer('all-distilroberta-v1')

def unload_model():
    return unload_current_model()

# Function to search the FAISS index
def search_index(aimodel, query, template, context_type='pdf', top_k=5):
    if model:
        model_handler = get_model_handler(aimodel)
    else:
        model_handler = get_model_handler()
        
    query_embedding = model.encode([query]).astype(np.float32)
    print(f"Searching context_type: {context_type}")
    if context_type == 'pdf':
        distances, indices = shared_state.index.search(query_embedding, top_k)
    else:
        distances, indices = shared_state.index_doc.search(query_embedding, top_k)
    results = []
    context = ""  # Initialize the context to pass to Mistral model
    for dist, idx in zip(distances[0], indices[0]):
        print(f"{idx} - {dist}")
        if context_type == 'pdf':
            text = shared_state.indexed_text_mapping[idx]
        else:
            text = shared_state.indexed_text_mapping_doc[idx]
        context += f"{text}\n"
        print(f"{text}")
        results.append({"index": int(idx), "distance": float(dist), "text": text})
        
    # Call the Mistral model to generate a response using the query and context
    # response = query_model(model_handler, query, context, template)
    # Apply reranking to the initial search results
     # Apply reranking to the initial search results with truncation
    reranked_results = sorted(
        results,
        key=lambda x: reranker(f"{query} [SEP] {truncate_text(query, x['text'])}")[0]['score'],
        reverse=True
    )
    
    # Generate a response using the top-ranked context
    top_context = "\n".join([truncate_text(query, res['text']) for res in reranked_results])
    response = query_model(model_handler, query, top_context, template)
    
    # Return both the search results and the response from the Mistral model
    return {
        "search_results": results,
        "model_response": response
    }