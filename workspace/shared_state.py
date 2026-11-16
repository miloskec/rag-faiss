import faiss

from rag_config import EMBEDDING_DIMENSION


FAISS_DIMENSION = EMBEDDING_DIMENSION
# Shared FAISS index and text mappings
index = faiss.IndexFlatL2(FAISS_DIMENSION)  # Initialize an empty FAISS index
indexed_text_mapping = []

index_doc = faiss.IndexFlatL2(FAISS_DIMENSION)  # Initialize an empty FAISS index
indexed_text_mapping_doc = []
