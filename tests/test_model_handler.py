import sys
import types

import pytest
import torch
from transformers import GenerationConfig

import base_model_handler
import model_utils
from base_model_handler import BaseModelHandler, ModelRuntimeError, inspect_quantization
from rag_config import MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, MODEL_ID


class FakeLinear4bit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = types.SimpleNamespace(bits=4, quant_state=object())


class FakeConfig:
    quantization_config = {
        "load_in_4bit": True,
        "quant_method": "bitsandbytes",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }


class FakeQwenModel:
    def __init__(self, generated_tokens=3):
        self.config = FakeConfig()
        self.generated_tokens = generated_tokens
        self.generate_kwargs = None
        self.generation_config = GenerationConfig(
            max_length=262144,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.8,
        )
        self._module = FakeLinear4bit()
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def modules(self):
        return iter([self._module])

    def parameters(self):
        return iter([self._parameter])

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        extra = torch.full(
            (1, self.generated_tokens),
            7,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, extra], dim=-1)


class QwenTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self):
        self.template_call = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = {"messages": messages, **kwargs}
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return "The supported answer is 2.4.0."


@pytest.fixture
def fake_bitsandbytes(monkeypatch):
    module = types.ModuleType("bitsandbytes")
    module.__version__ = "0.50.2"
    module.nn = types.SimpleNamespace(Linear4bit=FakeLinear4bit)
    monkeypatch.setitem(sys.modules, "bitsandbytes", module)
    return module


def test_inspect_quantization_requires_actual_4bit_modules(fake_bitsandbytes):
    report = inspect_quantization(FakeQwenModel())

    assert report["actual_4bit"] is True
    assert report["quant_type"] == "nf4"
    assert report["double_quant"] is True
    assert report["modules_4bit"] == 1
    assert report["weight_bits"] == 4


def test_initialize_loads_embedded_4bit_without_save(
    fake_bitsandbytes, monkeypatch
):
    calls = []
    tokenizer = QwenTokenizer()
    model = FakeQwenModel()

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("tokenizer", kwargs))
            return tokenizer

    class ModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append(("model", kwargs))
            return model

    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setattr(base_model_handler, "AutoTokenizer", TokenizerLoader)
    monkeypatch.setattr(base_model_handler, "AutoModelForCausalLM", ModelLoader)
    monkeypatch.setattr(base_model_handler.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(base_model_handler.torch.cuda, "synchronize", lambda: None)
    handler = BaseModelHandler("example/qwen-4bit")

    runtime = handler.initialize_model()

    model_call = next(call for call in calls if call[0] == "model")[1]
    assert model_call["device_map"] == {"": 0}
    assert model_call["dtype"] is torch.bfloat16
    assert model_call["attn_implementation"] == "sdpa"
    assert "quantization_config" not in model_call
    assert runtime["quantization"]["actual_4bit"] is True
    assert handler.model is model


def test_missing_cuda_fails_before_model_download(monkeypatch):
    monkeypatch.setattr(base_model_handler.torch.cuda, "is_available", lambda: False)

    with pytest.raises(ModelRuntimeError, match="NVIDIA CUDA GPU"):
        BaseModelHandler("example/qwen-4bit").initialize_model()


def test_native_chat_template_and_generation_settings(
    fake_bitsandbytes, monkeypatch
):
    tokenizer = QwenTokenizer()
    model = FakeQwenModel()
    handler = BaseModelHandler("example/qwen-4bit")
    handler.tokenizer = tokenizer
    handler.model = model
    handler.last_load_seconds = 1.5
    handler._load_metrics_pending = True
    monkeypatch.setattr(base_model_handler.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        base_model_handler.torch.cuda, "memory_allocated", lambda device: 1024**3
    )
    monkeypatch.setattr(
        base_model_handler.torch.cuda,
        "max_memory_allocated",
        lambda device: 2 * 1024**3,
    )

    result = handler.generate_grounded_response(
        "Which firmware?",
        "[Source: test; chunk: 1]\nFirmware 2.4.0 is required.",
    )

    assert tokenizer.template_call["tokenize"] is True
    assert tokenizer.template_call["add_generation_prompt"] is True
    assert tokenizer.template_call["return_dict"] is True
    assert tokenizer.template_call["truncation"] is True
    assert tokenizer.template_call["max_length"] == MAX_PROMPT_TOKENS
    generation_config = model.generate_kwargs["generation_config"]
    assert generation_config.max_new_tokens == MAX_NEW_TOKENS
    assert generation_config.do_sample is False
    assert generation_config.use_cache is True
    assert model.generate_kwargs["use_model_defaults"] is False
    assert result.metrics["model_load_ms"] == 1500.0
    assert result.metrics["input_tokens"] == 3
    assert result.metrics["generated_tokens"] == 3
    assert result.metrics["gpu_memory_allocated_mb"] == 1024.0


def test_model_handler_cache_reuses_single_model(monkeypatch):
    instances = []

    class CachedHandler:
        def __init__(self, model_id):
            self.model_id = model_id
            instances.append(self)

        def initialize_model(self):
            return self.runtime_info()

        def runtime_info(self):
            return {
                "model_id": self.model_id,
                "model_load_ms": 1.0,
                "quantization": {"actual_4bit": True},
            }

    monkeypatch.setattr(model_utils, "BaseModelHandler", CachedHandler)
    monkeypatch.setattr(
        model_utils,
        "model_cache",
        {"current_model": None, "model_id": None},
    )

    first = model_utils.get_model_handler()
    second = model_utils.get_model_handler("default")

    assert first is second
    assert len(instances) == 1
    assert first.model_id == MODEL_ID


def test_legacy_model_selection_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported model"):
        model_utils.get_model_handler("mistral-7b-instruct")
