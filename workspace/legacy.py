import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate

# Global variables for the model and tokenizer
model = None
tokenizer = None
save_dir = None
model_name = None
selected_model = None

def set_model():
    global save_dir, model_name, selected_model
    # Get the selected model from the environment variable
    selected_model = os.getenv('MODEL')
    print(f"Selected model: {selected_model}")
    if selected_model == 'mistral-7b-instruct':
        # Directory to save the model
        save_dir = "/models/saved_mistral_model"
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif selected_model == 'llama-7b-hf':
        # Directory to save the model
        save_dir = "/models/saved_llama_model"
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        raise ValueError(f"Unknown model selection: {selected_model}. Please set the 'MODEL' environment variable correctly.")

# Step 1: Check if the model and tokenizer files are saved locally
def is_model_saved(directory):
    if not directory:
        raise ValueError("Save directory is not set. Ensure the model setup is correct.")
    
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    return all(os.path.exists(os.path.join(directory, f)) for f in required_files)

# Initialize the model and tokenizer
def initialize_model():
    global model, tokenizer  # Declare them as global to access them in other functions
    
    set_model()  # Ensure save_dir and model_name are set
    
    try:
        if not is_model_saved(save_dir):
            print("Model not found locally. Downloading from Hugging Face and saving locally...")

            # Load model and tokenizer from Hugging Face and then save them
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            
            # Move model to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            if device == "cpu":
                print("CUDA not available, using CPU.")
            
            # Save the model and tokenizer locally
            tokenizer.save_pretrained(save_dir)
            model.save_pretrained(save_dir)
            print(f"Model and tokenizer saved locally to {save_dir}.")
        else:
            # If model is already saved locally, load from the saved directory
            print(f"Loading model and tokenizer from local directory: {save_dir}")
            tokenizer = AutoTokenizer.from_pretrained(save_dir)
            model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.float16)
            
            # Move model to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            if device == "cpu":
                print("CUDA not available, using CPU.")
    except Exception as e:
        print(f"Error loading or saving the model: {e}")
        raise

# Define the prompt format with the updated instruction
def generate_prompt(context, query, template):
    if template:
        return template.format(context=context, query=query)
    return f"""<s>[INST]Use the following context to answer the question. Do not repeat the context or provide instructions in the answer. {context} You should answer the following question:[/INST]{query}</s>"""

# Query the language model (LLM)
def query_llm(query, context, template):
    global model, tokenizer  # Use the global model and tokenizer
    
    # Check if the model and tokenizer are loaded properly
    if model is None or tokenizer is None:
        print("Model or tokenizer not initialized. Initializing now...")
        initialize_model()
        
    prompt = generate_prompt(context, query, template)
    
    # Add a pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        # Check if GPU is available and move inputs to GPU, otherwise use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Assign pad_token_id if necessary
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        start_index = inputs["input_ids"].shape[-1]
        output = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            max_new_tokens=1024, 
            temperature=0.4,
            top_k=50,
            top_p=0.93,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Decode and return the model's response
        response = tokenizer.decode(output[0][start_index:], skip_special_tokens=True)
        
        return response
    
    except torch.cuda.OutOfMemoryError:
        print("Out of GPU memory! Switching to CPU for inference.")
        # Move the model to CPU and try again
        model = model.to("cpu")
        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        
        start_index = inputs["input_ids"].shape[-1]
        # Retry generation on CPU
        output = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            max_new_tokens=1024, 
            temperature=0.4,
            top_k=50,
            top_p=0.93,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        response = tokenizer.decode(output[0][start_index:], skip_special_tokens=True)
        return response

    except Exception as e:
        print(f"Error during model inference: {e}")
        raise

# Initialize the model and tokenizer at the start of the application
initialize_model()