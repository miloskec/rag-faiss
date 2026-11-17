import gc
import threading

from rag_config import MODEL_ID

import torch

from base_model_handler import BaseModelHandler, GenerationResult
from prompting import generate_prompt


model_cache = {
    "current_model": None,
    "model_id": None,
}
_model_lock = threading.Lock()


def resolve_model_selection(selected_model=None):
    requested = (selected_model or "").strip()
    if not requested or requested in {"default", MODEL_ID}:
        return MODEL_ID
    raise ValueError(
        f"Unsupported model '{requested}'. This deployment is configured to use "
        f"only MODEL_ID={MODEL_ID}."
    )


def unload_current_model():
    with _model_lock:
        handler = model_cache["current_model"]
        if handler is not None:
            if hasattr(handler, "model"):
                handler.model = None
            if hasattr(handler, "tokenizer"):
                handler.tokenizer = None
            model_cache["current_model"] = None
            model_cache["model_id"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"status": "unloaded"}


def get_model_handler(selected_model=None):
    model_id = resolve_model_selection(selected_model)
    with _model_lock:
        if model_cache["model_id"] == model_id and model_cache["current_model"]:
            return model_cache["current_model"]

        handler = BaseModelHandler(model_id)
        handler.initialize_model()
        model_cache["current_model"] = handler
        model_cache["model_id"] = model_id
        return handler


def get_model_runtime_info():
    handler = model_cache["current_model"]
    return handler.runtime_info() if handler is not None else None


def query_model(model_handler, query, context, template=None):
    if model_handler is None:
        raise ValueError("A valid initialized model handler is required")
    if hasattr(model_handler, "generate_grounded_response"):
        return model_handler.generate_grounded_response(query, context, template)
    if not hasattr(model_handler, "generate_response"):
        raise ValueError("A valid initialized model handler is required")

    prompt = generate_prompt(context, query, template)
    text = model_handler.generate_response(prompt)
    return GenerationResult(
        text=text,
        metrics={
            "model_load_ms": 0.0,
            "prompt_processing_ms": 0.0,
            "generation_ms": 0.0,
            "input_tokens": 0,
            "generated_tokens": 0,
            "generation_tokens_per_second": 0.0,
            "gpu_memory_allocated_mb": 0.0,
            "gpu_peak_memory_mb": 0.0,
        },
    )
