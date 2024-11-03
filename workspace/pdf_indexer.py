import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import shared_state  # Import the shared index and mappings from shared_state

# Initialize the sentence transformer model (to convert text to embeddings)
model = SentenceTransformer('all-distilroberta-v1')

# FAISS index file and text mapping paths
FAISS_INDEX_PATH = "./workspace/data/pdf/index.faiss"
TEXT_MAPPING_PATH = "./workspace/data/pdf/text_mapping.pkl"

# Initialize FAISS index
index = faiss.IndexFlatL2(768)
indexed_text_mapping = []
# Save FAISS index to disk
def save_faiss_index():
    faiss.write_index(shared_state.index, FAISS_INDEX_PATH)

# Save text mappings to disk
def save_mappings():
    with open(TEXT_MAPPING_PATH, 'wb') as f:
        pickle.dump(shared_state.indexed_text_mapping, f)

# Function to load FAISS index from disk
def load_faiss_index(embedding_size=768):
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        return faiss.IndexFlatL2(embedding_size)

# Function to load text mappings from disk
def load_mappings():
    if os.path.exists(TEXT_MAPPING_PATH):
        with open(TEXT_MAPPING_PATH, 'rb') as f:
            shared_state.indexed_text_mapping = pickle.load(f)
            
# Function to extract sections from PDFs
def extract_sections_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    current_heading = "Introduction"
    current_content = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if is_heading(span):
                            if current_heading and current_content:
                                sections.append({"title": current_heading, "text": current_content.strip()})
                                current_content = ""
                            current_heading = text
                        else:
                            current_content += " " + text
    if current_heading and current_content:
        sections.append({"title": current_heading, "text": current_content.strip()})

    return sections

# Helper function to detect if a block is a heading based on font size and style
def is_heading(span):
    font_size = span["size"]
    is_bold = span["flags"] & 1
    return font_size > 12 or is_bold

# Function to add PDFs to the FAISS index
def update_pdf_index(pdf_folder):
    global indexed_text_mapping
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            sections = extract_sections_from_pdf(file_path)
            for section in sections:
                chunk_text = f"{section['title']} - {section['text']}"
                description = f"{section['title']} - {section['text']} [{filename}]"
                chunk_embedding = model.encode([chunk_text])
                shared_state.index.add(np.array(chunk_embedding).astype(np.float32))
                shared_state.indexed_text_mapping.append(description)

# Function to clear index and reset mappings
def clear_index():
    shared_state.index.reset()
    shared_state.indexed_text_mapping.clear()

# Main function to initialize FAISS and load existing data
def initialize():
    # Initialize FAISS index and load data into shared state
    shared_state.index = load_faiss_index(embedding_size=768)
    load_mappings()

# Function to save index and mappings after updates
def save_index_and_mappings():
    save_faiss_index()  # Save FAISS index to disk
    save_mappings()     # Save text mappings to disk