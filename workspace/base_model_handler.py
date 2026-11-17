import copy
import logging
import time
from dataclasses import dataclass

from rag_config import (
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    MODEL_ID,
    env_flag,
    get_hf_token,
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompting import build_chat_messages, generate_prompt


class ModelRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    text: str
    metrics: dict


def _quantization_value(quantization_config, name, default=None):
    if isinstance(quantization_config, dict):
        return quantization_config.get(name, default)
    return getattr(quantization_config, name, default)


def inspect_quantization(model):
    try:
        import bitsandbytes as bnb

        bnb_version = bnb.__version__
        modules_4bit = [
            module
            for module in model.modules()
            if isinstance(module, bnb.nn.Linear4bit)
        ]
    except ImportError:
        bnb_version = None
        modules_4bit = []

    quantization_config = getattr(model.config, "quantization_config", None)
    weight = modules_4bit[0].weight if modules_4bit else None
    load_in_4bit = bool(_quantization_value(quantization_config, "load_in_4bit", False))
    quant_type = _quantization_value(quantization_config, "bnb_4bit_quant_type")
    weight_bits = getattr(weight, "bits", None)
    quant_state = getattr(weight, "quant_state", None)

    return {
        "actual_4bit": bool(modules_4bit) and load_in_4bit and quant_type == "nf4",
        "load_in_4bit": load_in_4bit,
        "quant_method": _quantization_value(quantization_config, "quant_method"),
        "quant_type": quant_type,
        "double_quant": bool(
            _quantization_value(
                quantization_config, "bnb_4bit_use_double_quant", False
            )
        ),
        "compute_dtype": str(
            _quantization_value(quantization_config, "bnb_4bit_compute_dtype")
        ),
        "modules_4bit": len(modules_4bit),
        "weight_class": type(weight).__name__ if weight is not None else None,
        "weight_bits": int(weight_bits) if weight_bits is not None else None,
        "quant_state_loaded": quant_state is not None,
        "bitsandbytes_version": bnb_version,
    }


class BaseModelHandler:
    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.last_load_seconds = None
        self.quantization = None
        self._load_metrics_pending = False

    def initialize_model(self):
        if not torch.cuda.is_available():
            raise ModelRuntimeError(
                "The selected bitsandbytes Qwen model requires an NVIDIA CUDA GPU. "
                "Verify the NVIDIA driver and Docker GPU access."
            )
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise ModelRuntimeError(
                "bitsandbytes is required for the prequantized Qwen model"
            ) from error

        token = get_hf_token()
        local_only = env_flag("MODEL_LOCAL_FILES_ONLY", default=False)
        started = time.perf_counter()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=token,
                local_files_only=local_only,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=token,
                local_files_only=local_only,
                dtype=torch.bfloat16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                use_safetensors=True,
            )
            self.model.eval()
            torch.cuda.synchronize()
        except Exception as error:
            raise ModelRuntimeError(
                f"Failed to load 4-bit model '{self.model_id}': {error}"
            ) from error

        self.quantization = inspect_quantization(self.model)
        if not self.quantization["actual_4bit"]:
            raise ModelRuntimeError(
                "The model loaded, but no genuine bitsandbytes NF4 4-bit modules "
                f"were detected: {self.quantization}"
            )
        self.last_load_seconds = time.perf_counter() - started
        self._load_metrics_pending = True
        logging.info(
            "Loaded %s in %.2fs with actual_4bit=%s quant_type=%s double_quant=%s",
            self.model_id,
            self.last_load_seconds,
            self.quantization["actual_4bit"],
            self.quantization["quant_type"],
            self.quantization["double_quant"],
        )
        return self.runtime_info()

    def runtime_info(self):
        return {
            "model_id": self.model_id,
            "model_load_ms": (
                round(self.last_load_seconds * 1000, 2)
                if self.last_load_seconds is not None
                else None
            ),
            "quantization": self.quantization,
        }

    def _build_inputs(self, query, context, template=None):
        if template:
            prompt = generate_prompt(context, query, template)
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PROMPT_TOKENS,
                add_special_tokens=True,
            )
            return dict(encoded)

        messages = build_chat_messages(query, context)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )
        if isinstance(encoded, torch.Tensor):
            encoded = {"input_ids": encoded}
        return dict(encoded)

    def generate_grounded_response(
        self,
        query,
        context,
        template=None,
        max_new_tokens=MAX_NEW_TOKENS,
    ):
        if self.model is None or self.tokenizer is None:
            raise ModelRuntimeError("The Qwen model has not been initialized")

        started = time.perf_counter()
        inputs = self._build_inputs(query, context, template)
        prompt_processing_seconds = time.perf_counter() - started
        input_tokens = int(inputs["input_ids"].shape[-1])
        input_device = next(self.model.parameters()).device
        inputs = {
            name: value.to(input_device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.use_cache = True
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                generation_config=generation_config,
                use_model_defaults=False,
            )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started

        generated_ids = output[0, input_tokens:]
        generated_tokens = int(generated_ids.shape[-1])
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        self.quantization = inspect_quantization(self.model)
        tokens_per_second = (
            generated_tokens / generation_seconds if generation_seconds > 0 else 0.0
        )
        metrics = {
            "model_load_ms": (
                round((self.last_load_seconds or 0.0) * 1000, 2)
                if self._load_metrics_pending
                else 0.0
            ),
            "prompt_processing_ms": round(prompt_processing_seconds * 1000, 2),
            "generation_ms": round(generation_seconds * 1000, 2),
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "generation_tokens_per_second": round(tokens_per_second, 2),
            "gpu_memory_allocated_mb": round(
                torch.cuda.memory_allocated(input_device) / 1024**2, 2
            ),
            "gpu_peak_memory_mb": round(
                torch.cuda.max_memory_allocated(input_device) / 1024**2, 2
            ),
        }
        self._load_metrics_pending = False
        logging.info(
            "Qwen generated %d tokens in %.2fs (%.2f tokens/s, %d input tokens)",
            generated_tokens,
            generation_seconds,
            tokens_per_second,
            input_tokens,
        )
        return GenerationResult(text=text, metrics=metrics)
