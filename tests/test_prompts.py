import pytest

from model_utils import generate_prompt


@pytest.mark.parametrize(
    ("style", "opening"),
    [
        ("llama2", "<s>[INST]"),
        ("mistral", "[INST]"),
        ("chatml", "<|im_start|>system"),
        ("sambalingo", "<|system|>"),
        ("plain", "Answer the question"),
    ],
)
def test_supported_prompt_styles_preserve_question(style, opening):
    prompt = generate_prompt(
        "Aster-100 requires firmware 2.4.0.",
        "Which firmware is required?",
        prompt_style=style,
    )

    assert prompt.startswith(opening)
    assert "Question: Which firmware is required?" in prompt
    assert "2.4.0" in prompt
    assert prompt.index("Question:") < prompt.index("Context:")


def test_custom_template_receives_context_and_query():
    prompt = generate_prompt(
        "context text",
        "query text",
        template="Q={query}\nC={context}",
    )

    assert prompt == "Q=query text\nC=context text"


def test_unknown_template_placeholder_fails():
    with pytest.raises(ValueError, match="unsupported placeholder"):
        generate_prompt("context", "query", template="{missing}")
