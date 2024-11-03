import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

class BaseModelHandler:
    def __init__(self, model_name, save_dir):
        self.model_name = model_name
        self.save_dir = save_dir
        self.model = None
        self.tokenizer = None

    def is_model_saved(self):
        """Check if the model and tokenizer are saved locally."""
        try:
            required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            return all(os.path.exists(os.path.join(self.save_dir, f)) for f in required_files)
        except Exception as e:
            print(f"Error checking if model is saved: {e}")
            logging.error(f"Error checking if model is saved: {e}")
            return False

    def initialize_model(self):
        """Initialize and load the model and tokenizer."""
        try:
            if not self.is_model_saved():
                print(f"Model {self.model_name} not found locally. Downloading from Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
                
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
                self.model = AutoModelForCausalLM.from_pretrained(self.save_dir, torch_dtype=torch.float16)
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

    def generate_response(self, prompt, max_new_tokens=1024, temperature=0.4, top_k=50, top_p=0.93):
        """Generate a response from the model given a prompt."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
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
            logging.error(f"Error checking if model is saved: {e}")
            raise
