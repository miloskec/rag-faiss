from base_model_handler import BaseModelHandler

class MistralModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__(
            model_name="mistralai/Mistral-7B-Instruct-v0.3", 
            save_dir="/models/saved_mistral_model"
        )

# Instantiate and initialize Mistral model
#mistral_handler = MistralModelHandler()
#mistral_handler.initialize_model()
