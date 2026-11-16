import torch

import base_model_handler
from base_model_handler import BaseModelHandler
from model_utils import query_model
from rag_config import MAX_PROMPT_TOKENS


class RecordingTokenizer:
    pad_token_id = 0
    chat_template = "chat-template"
    last_call = None

    def __call__(self, prompt, **kwargs):
        self.last_call = {"prompt": prompt, **kwargs}
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
        return (
            f"SYSTEM={messages[0]['content']}\n"
            f"USER={messages[1]['content']}\nASSISTANT="
        )

    def decode(self, tokens, skip_special_tokens=True):
        return "test response"


class EchoModel:
    def generate(self, input_ids, **kwargs):
        return input_ids


class SavingTokenizer(RecordingTokenizer):
    def __init__(self, calls):
        self.calls = calls
        self.pad_token = None
        self.eos_token = "</s>"

    def save_pretrained(self, path):
        self.calls.append(("tokenizer_save", path))


class SavingModel:
    def __init__(self, calls):
        self.calls = calls

    def to(self, device):
        return self

    def save_pretrained(self, path):
        self.calls.append(("model_save", path))


class BrokenTemplateTokenizer:
    chat_template = "broken"

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("system role is unsupported")


def test_chat_template_is_preferred_when_available():
    handler = BaseModelHandler("fake", "/tmp/fake")
    handler.tokenizer = RecordingTokenizer()

    prepared = handler.prepare_prompt(
        "Which firmware is required?",
        "Aster-100 requires firmware 2.4.0.",
    )

    assert prepared.text.startswith("SYSTEM=")
    assert "Question: Which firmware is required?" in prepared.text
    assert "2.4.0" in prepared.text
    assert prepared.add_special_tokens is False


def test_broken_chat_template_uses_fallback_prompt():
    handler = BaseModelHandler("fake", "/tmp/fake", prompt_style="mistral")
    handler.tokenizer = BrokenTemplateTokenizer()

    prepared = handler.prepare_prompt("question", "context")

    assert prepared.text.startswith("[INST]")
    assert "Question: question" in prepared.text
    assert prepared.add_special_tokens is True


def test_generation_uses_configured_prompt_token_limit():
    handler = BaseModelHandler("fake", "/tmp/fake")
    handler.tokenizer = RecordingTokenizer()
    handler.model = EchoModel()
    prompt = handler.format_prompt("question", "context")

    handler.generate_response(prompt)

    assert handler.tokenizer.last_call["max_length"] == MAX_PROMPT_TOKENS
    assert handler.tokenizer.last_call["truncation"] is True


def test_query_model_disables_special_tokens_for_chat_template():
    handler = BaseModelHandler("fake", "/tmp/fake")
    handler.tokenizer = RecordingTokenizer()
    handler.model = EchoModel()

    query_model(handler, "question", "context")

    assert handler.tokenizer.last_call["add_special_tokens"] is False


def test_model_loading_passes_hf_token_explicitly(tmp_path, monkeypatch):
    calls = []
    tokenizer = SavingTokenizer(calls)
    model = SavingModel(calls)

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

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(base_model_handler, "AutoTokenizer", TokenizerLoader)
    monkeypatch.setattr(base_model_handler, "AutoModelForCausalLM", ModelLoader)
    monkeypatch.setattr(base_model_handler.torch.cuda, "is_available", lambda: False)
    handler = BaseModelHandler("example/model", tmp_path)

    handler.initialize_model()

    tokenizer_call = next(call for call in calls if call[0] == "tokenizer")[1]
    model_call = next(call for call in calls if call[0] == "model")[1]
    assert tokenizer_call["token"] == "test-token"
    assert model_call["token"] == "test-token"
