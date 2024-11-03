import os
import torch
from llama_module import LlamaModelHandler
from mistral_module import MistralModelHandler
import logging

# Global variable to keep track of the currently loaded model and its name
model_cache = {
    'current_model': None,
    'model_name': None
}

def unload_current_model():
    try:
        """Unloads the current model from memory to free up resources."""
        if model_cache['current_model'] is not None:
            print(f"Unloading model: {model_cache['model_name']}")
            del model_cache['current_model']
            torch.cuda.empty_cache()  # Clear the GPU memory
            model_cache['current_model'] = None
            model_cache['model_name'] = None
    except Exception as e:
        logging.error(f"Error unloading model: {str(e)}")
        return {"error": str(e)}
    return {"status": "unloaded"}

def get_model_handler(selected_model=None):
    try:
        """Dynamically select the model and cache it to avoid reloading unnecessarily."""
        global model_cache

        if not selected_model:
            selected_model = os.getenv('MODEL', 'llama-7b-hf')  # Default to LLaMA

        if selected_model == model_cache['model_name']:
            # If the selected model is already loaded, return it
            print(f"Using cached model: {model_cache['model_name']}")
            return model_cache['current_model']

        # If another model is selected, unload the current model
        unload_current_model()

        # Load the new model
        if selected_model == 'llama-7b-hf':
            handler = LlamaModelHandler()
        elif selected_model == 'mistral-7b-instruct':
            handler = MistralModelHandler()
        else:
            raise ValueError(f"Unknown model selection: {selected_model}")

        # Initialize and cache the new model
        handler.initialize_model()
        model_cache['current_model'] = handler
        model_cache['model_name'] = selected_model

        return handler
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return {"error": str(e)}

def generate_prompt(context, query, template=None):
    """Generate the prompt for the model."""
    if template:
        return template.format(context=context, query=query)
    return f"""<s>[INST]Use the following context to answer the question. Do not repeat the context or provide instructions in the answer. {context} You should answer the following question:[/INST]{query}</s>"""

def query_model(model_handler, query, context, template=None):
    """Generate the response using the provided model handler."""
    prompt = generate_prompt(context, query, template)
    return model_handler.generate_response(prompt)
