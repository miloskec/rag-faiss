import pytest

from prompting import build_chat_messages, generate_prompt


def test_native_chat_messages_preserve_question_and_context():
    messages = build_chat_messages(
        "Which firmware is required?",
        "Aster-100 requires firmware 2.4.0.",
    )
    user_content = messages[1]["content"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Question: Which firmware is required?" in user_content
    assert "2.4.0" in user_content
    assert user_content.index("Question:") < user_content.index("Context:")


def test_custom_template_receives_context_and_query():
    prompt = generate_prompt(
        "context text",
        "query text",
        template="Q={query}\nC={context}",
    )

    assert prompt == "Q=query text\nC=context text"


def test_default_prompt_is_grounded_user_content():
    prompt = generate_prompt("context text", "query text")

    assert prompt.startswith("Answer the question using only the supplied context.")
    assert "Question: query text" in prompt
    assert prompt.index("Question:") < prompt.index("Context:")


def test_unknown_template_placeholder_fails():
    with pytest.raises(ValueError, match="unsupported placeholder"):
        generate_prompt("context", "query", template="{missing}")
