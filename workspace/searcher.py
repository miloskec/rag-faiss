import numpy as np
from sentence_transformers import SentenceTransformer
import shared_state  # Import the shared index and mappings

# Dynamic import based on the selected model
from model_utils import get_model_handler, unload_current_model # Function to dynamically select the model
from model_utils import query_model  # Function to query the selected model

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
    response = query_model(model_handler, query, context, template)
    
    # Return both the search results and the response from the Mistral model
    return {
        "search_results": results,
        "model_response": response
    }