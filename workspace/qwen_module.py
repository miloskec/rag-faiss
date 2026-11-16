from base_model_handler import BaseModelHandler

class QwenModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            save_dir="/models/saved_qwen_model",
            prompt_style="chatml",
        )

# Instantiate and initialize Qwen model
#qwen_handler = QwenModelHandler()
#qwen_handler.initialize_model()
