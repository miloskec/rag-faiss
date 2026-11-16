import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from prompting import PreparedPrompt, build_chat_messages, generate_prompt
from rag_config import MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, env_flag

class BaseModelHandler:
    def __init__(self, model_name, save_dir, prompt_style="plain"):
        self.model_name = model_name
        self.save_dir = save_dir
        self.prompt_style = prompt_style
        self.model = None
        self.tokenizer = None

    def is_model_saved(self):
        """Check if the model and tokenizer are saved locally."""
        if not os.path.isdir(self.save_dir):
            return False
        config_exists = os.path.exists(os.path.join(self.save_dir, "config.json"))
        tokenizer_exists = any(
            os.path.exists(os.path.join(self.save_dir, name))
            for name in ("tokenizer.json", "tokenizer.model")
        )
        weights_exist = any(
            os.path.exists(os.path.join(self.save_dir, name))
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        )
        return config_exists and tokenizer_exists and weights_exist

    def initialize_model(self):
        """Initialize and load the model and tokenizer."""
        try:
            if not self.is_model_saved():
                print(f"Model {self.model_name} not found locally. Downloading from Hugging Face...")
                local_only = env_flag("MODEL_LOCAL_FILES_ONLY", default=False)
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=local_only
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    local_files_only=local_only,
                )
                
                # Add a padding token if it does not exist
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        self.model.resize_token_embeddings(len(self.tokenizer))
                        
                # Save model and tokenizer locally
                self.tokenizer.save_pretrained(self.save_dir)
                self.model.save_pretrained(self.save_dir)
                print(f"Model and tokenizer saved locally to {self.save_dir}.")
            else:
                # Load model and tokenizer from local directory
                print(f"Loading model {self.model_name} from local directory: {self.save_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.save_dir, torch_dtype=dtype
                )
                # Add a padding token if it does not exist
                if self.tokenizer.pad_token is None:
                    if self.tokenizer.eos_token:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Move model to GPU if available, otherwise use CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            print(f"Using device: {device}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            logging.error(f"Error checking if model is saved: {e}")
            raise

    def prepare_prompt(self, query, context, template=None):
        if template:
            return PreparedPrompt(
                generate_prompt(context, query, template, self.prompt_style),
                add_special_tokens=True,
            )
        if self.tokenizer is not None and getattr(self.tokenizer, "chat_template", None):
            try:
                return PreparedPrompt(
                    self.tokenizer.apply_chat_template(
                        build_chat_messages(query, context),
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                    add_special_tokens=False,
                )
            except (TypeError, ValueError) as error:
                logging.warning(
                    "Tokenizer chat template failed (%s); using %s fallback",
                    error,
                    self.prompt_style,
                )
        return PreparedPrompt(
            generate_prompt(context, query, prompt_style=self.prompt_style),
            add_special_tokens=self.prompt_style in {"plain", "mistral"},
        )

    def format_prompt(self, query, context, template=None):
        """Return prompt text for callers that do not need tokenizer flags."""
        return self.prepare_prompt(query, context, template).text

    def generate_response(
        self,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.4,
        top_k=50,
        top_p=0.93,
        add_special_tokens=True,
    ):
        """Generate a response from the model given a prompt."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_TOKENS,
                add_special_tokens=add_special_tokens,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {key: value.to(device) for key, value in inputs.items()}

            start_index = inputs["input_ids"].shape[-1]
            output = self.model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            return self.tokenizer.decode(output[0][start_index:], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            print("Out of GPU memory! Switching to CPU for inference.")
            logging.error("Out of GPU memory! Switching to CPU for inference.")
            self.model = self.model.to("cpu")
            inputs = {key: value.to("cpu") for key, value in inputs.items()}
            output = self.model.generate(
                inputs['input_ids'], 
                attention_mask=inputs['attention_mask'], 
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            return self.tokenizer.decode(output[0][start_index:], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during model inference: {e}")
            logging.error("Error during model inference: %s", e)
            raise
