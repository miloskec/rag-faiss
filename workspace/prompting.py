from dataclasses import dataclass


GROUNDING_INSTRUCTION = (
    "Answer the question using only the supplied context. "
    "If the context is insufficient, say that the answer is not available. "
    "Do not invent facts or repeat these instructions."
)


@dataclass(frozen=True)
class PreparedPrompt:
    text: str
    add_special_tokens: bool


def build_user_content(query, context):
    query = (query or "").strip()
    context = (context or "").strip()
    if not query:
        raise ValueError("A non-empty query is required")
    return f"{GROUNDING_INSTRUCTION}\n\nQuestion: {query}\n\nContext:\n{context}"


def build_chat_messages(query, context):
    return [
        {
            "role": "system",
            "content": "You are a grounded retrieval assistant.",
        },
        {"role": "user", "content": build_user_content(query, context)},
    ]


def generate_prompt(context, query, template=None, prompt_style="llama2"):
    """Build a prompt with the question before context so truncation preserves it."""
    query = (query or "").strip()
    context = (context or "").strip()
    if not query:
        raise ValueError("A non-empty query is required")
    if template:
        try:
            return template.format(context=context, query=query)
        except KeyError as error:
            raise ValueError(
                f"Prompt template contains unsupported placeholder: {error.args[0]}"
            ) from error

    body = build_user_content(query, context)
    if prompt_style == "llama2":
        return f"<s>[INST] {body} [/INST]"
    if prompt_style == "mistral":
        return f"[INST] {body} [/INST]"
    if prompt_style == "chatml":
        return (
            "<|im_start|>system\n"
            "You are a grounded retrieval assistant.<|im_end|>\n"
            f"<|im_start|>user\n{body}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if prompt_style == "sambalingo":
        return (
            "<|system|>\nYou are a grounded retrieval assistant.</s>\n"
            f"<|user|>\n{body}</s>\n"
            "<|assistant|>"
        )
    if prompt_style == "plain":
        return body
    raise ValueError(f"Unsupported prompt style: {prompt_style}")
