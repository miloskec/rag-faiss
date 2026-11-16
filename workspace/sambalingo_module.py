from base_model_handler import BaseModelHandler

class SambalingoModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__(
            model_name="sambanovasystems/SambaLingo-Serbian-Chat",
            save_dir="/models/saved_samba_model",
            prompt_style="sambalingo",
        )
