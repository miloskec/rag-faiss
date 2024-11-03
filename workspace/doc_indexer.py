import os
import fitz  # PyMuPDF for PDF processing
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import docx  # Python-docx for Word processing
from odf.opendocument import load  # ODFpy for ODT processing
from odf.text import P
import shared_state  # Import the shared index and mappings from shared_state
import logging

# Initialize the sentence transformer model (to convert text to embeddings)
model = SentenceTransformer('all-distilroberta-v1')

# FAISS index file and text mapping paths
FAISS_INDEX_PATH = "./workspace/data/doc/index.faiss"
TEXT_MAPPING_PATH = "./workspace/data/doc/text_mapping.pkl"

# Initialize FAISS index
index_doc = faiss.IndexFlatL2(768)
indexed_text_mapping_doc = []

# Save FAISS index to disk
def save_faiss_index():
    faiss.write_index(shared_state.index_doc, FAISS_INDEX_PATH)

# Save text mappings to disk
def save_mappings():
    with open(TEXT_MAPPING_PATH, 'wb') as f:
        pickle.dump(shared_state.indexed_text_mapping_doc, f)

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
            shared_state.indexed_text_mapping_doc = pickle.load(f)

# Function to extract sections from Word documents
def extract_sections_from_word(docx_path):
    try:    
        doc = docx.Document(docx_path)
        sections = []
        current_heading = "Introduction"
        current_content = ""

        for para in doc.paragraphs:
            if is_heading_word(para):
                if current_heading and current_content:
                    sections.append({"title": current_heading, "text": current_content.strip()})
                    current_content = ""
                current_heading = para.text
            else:
                current_content += " " + para.text

        if current_heading and current_content:
            sections.append({"title": current_heading, "text": current_content.strip()})

        return sections
    except Exception as e:
        logging.error(f"Error processing Word document: {docx_path}")
        logging.error(str(e))
        return []

# Function to extract sections from ODT files
def extract_sections_from_odt(odt_path):
    try:
        doc = load(odt_path)
        sections = []
        current_heading = "Introduction"
        current_content = ""

        for paragraph in doc.getElementsByType(P):
            para_text = "".join([node.data for node in paragraph.childNodes if hasattr(node, 'data')])
            if is_heading_odt(paragraph):
                if current_heading and current_content:
                    sections.append({"title": current_heading, "text": current_content.strip()})
                    current_content = ""
                current_heading = para_text
            else:
                current_content += " " + para_text

        if current_heading and current_content:
            sections.append({"title": current_heading, "text": current_content.strip()})

        return sections
    except Exception as e:
        logging.error(f"Error processing ODT file: {odt_path}")
        logging.error(str(e))
        return []

# Helper function to detect if a paragraph is a heading based on style in Word documents
def is_heading_word(para):
    return para.style.name.startswith('Heading')

# Helper function to detect if a paragraph is a heading in ODT files (basic check)
def is_heading_odt(para):
    return para.attributes.get('text:style-name', '').startswith('Heading')

# Function to add Word or ODT files to the FAISS index
def update_doc_index(doc_folder):
    try:
        global indexed_text_mapping_doc
        for filename in os.listdir(doc_folder):
            file_path = os.path.join(doc_folder, filename)

            if filename.endswith(".docx"):
                sections = extract_sections_from_word(file_path)
            elif filename.endswith(".odt"):
                sections = extract_sections_from_odt(file_path)
            else:
                print(f"Unsupported file format: {filename}")
                continue

            for section in sections:
                chunk_text = f"{section['title']} - {section['text']}"
                description = f"{section['title']} - {section['text']} [{filename}]"
                chunk_embedding = model.encode([chunk_text])
                shared_state.index_doc.add(np.array(chunk_embedding).astype(np.float32))
                shared_state.indexed_text_mapping_doc.append(description)
    except Exception as e:
        logging.error(f"Error processing document: {file_path}")
        logging.error(str(e))
        
# Function to clear index and reset mappings
def clear_index():
    shared_state.index_doc.reset()
    shared_state.indexed_text_mapping_doc.clear()

# Main function to initialize FAISS and load existing data
def initialize():
    # Initialize FAISS index and load data into shared state
    shared_state.index_doc = load_faiss_index(embedding_size=768)
    load_mappings()

# Function to save index and mappings after updates
def save_index_and_mappings():
    save_faiss_index()  # Save FAISS index to disk
    save_mappings()     # Save text mappings to disk
