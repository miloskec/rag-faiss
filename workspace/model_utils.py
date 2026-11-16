import gc
import logging
import threading

from rag_config import DEFAULT_MODEL

import torch

from llama_module import LlamaModelHandler
from mistral_module import MistralModelHandler
from prompting import generate_prompt
from qwen_module import QwenModelHandler
from sambalingo_module import SambalingoModelHandler


MODEL_HANDLERS = {
    "llama-7b-hf": LlamaModelHandler,
    "mistral-7b-instruct": MistralModelHandler,
    "qwen2.5-7b-instruct": QwenModelHandler,
    "sambalingo-chat": SambalingoModelHandler,
}

model_cache = {"current_model": None, "model_name": None}
_model_lock = threading.Lock()


def unload_current_model():
    with _model_lock:
        handler = model_cache["current_model"]
        if handler is not None:
            logging.info("Unloading model: %s", model_cache["model_name"])
            if hasattr(handler, "model"):
                handler.model = None
            if hasattr(handler, "tokenizer"):
                handler.tokenizer = None
            model_cache["current_model"] = None
            model_cache["model_name"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"status": "unloaded"}


def get_model_handler(selected_model=None):
    """Return an initialized handler, raising an explicit error on invalid input."""
    selected_model = (selected_model or DEFAULT_MODEL).strip().lower()
    with _model_lock:
        if selected_model == model_cache["model_name"]:
            return model_cache["current_model"]

        handler_class = MODEL_HANDLERS.get(selected_model)
        if handler_class is None:
            supported = ", ".join(sorted(MODEL_HANDLERS))
            raise ValueError(
                f"Unknown model selection: {selected_model}. Supported models: {supported}"
            )

        if model_cache["current_model"] is not None:
            previous = model_cache["current_model"]
            previous.model = None
            previous.tokenizer = None
            model_cache["current_model"] = None
            model_cache["model_name"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        handler = handler_class()
        handler.initialize_model()
        model_cache["current_model"] = handler
        model_cache["model_name"] = selected_model
        return handler


def query_model(model_handler, query, context, template=None):
    if model_handler is None or not hasattr(model_handler, "generate_response"):
        raise ValueError("A valid initialized model handler is required")
    if hasattr(model_handler, "prepare_prompt"):
        prepared = model_handler.prepare_prompt(query, context, template)
        return model_handler.generate_response(
            prepared.text,
            add_special_tokens=prepared.add_special_tokens,
        )
    if hasattr(model_handler, "format_prompt"):
        prompt = model_handler.format_prompt(query, context, template)
    else:
        prompt_style = getattr(model_handler, "prompt_style", "plain")
        prompt = generate_prompt(context, query, template, prompt_style)
    return model_handler.generate_response(prompt)
