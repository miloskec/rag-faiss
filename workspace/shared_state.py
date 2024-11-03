import faiss

# Shared FAISS index and text mappings
index = faiss.IndexFlatL2(768)  # Initialize an empty FAISS index
indexed_text_mapping = []       # Shared list to hold text mappings

index_doc = faiss.IndexFlatL2(768)  # Initialize an empty FAISS index
indexed_text_mapping_doc = []       # Shared list to hold text mappings