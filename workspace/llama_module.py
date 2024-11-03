from base_model_handler import BaseModelHandler

class LlamaModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Llama-2-7b-chat-hf", 
            save_dir="/models/saved_llama_model"
        )

# Instantiate and initialize LLaMA model
#llama_handler = LlamaModelHandler()
#llama_handler.initialize_model()
