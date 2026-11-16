import torch

from base_model_handler import BaseModelHandler
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


class BrokenTemplateTokenizer:
    chat_template = "broken"

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("system role is unsupported")


def test_chat_template_is_preferred_when_available():
    handler = BaseModelHandler("fake", "/tmp/fake")
    handler.tokenizer = RecordingTokenizer()

    prompt = handler.format_prompt(
        "Which firmware is required?",
        "Aster-100 requires firmware 2.4.0.",
    )

    assert prompt.startswith("SYSTEM=")
    assert "Question: Which firmware is required?" in prompt
    assert "2.4.0" in prompt


def test_broken_chat_template_uses_fallback_prompt():
    handler = BaseModelHandler("fake", "/tmp/fake", prompt_style="mistral")
    handler.tokenizer = BrokenTemplateTokenizer()

    prompt = handler.format_prompt("question", "context")

    assert prompt.startswith("[INST]")
    assert "Question: question" in prompt


def test_generation_uses_configured_prompt_token_limit():
    handler = BaseModelHandler("fake", "/tmp/fake")
    handler.tokenizer = RecordingTokenizer()
    handler.model = EchoModel()
    prompt = handler.format_prompt("question", "context")

    handler.generate_response(prompt)

    assert handler.tokenizer.last_call["max_length"] == MAX_PROMPT_TOKENS
    assert handler.tokenizer.last_call["truncation"] is True
