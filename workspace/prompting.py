GROUNDING_INSTRUCTION = (
    "Answer the question using only the supplied context. "
    "If the context is insufficient, say that the answer is not available. "
    "Do not invent facts or repeat these instructions."
)


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


def generate_prompt(context, query, template=None):
    """Build custom prompt text; Qwen otherwise uses its native chat template."""
    if template:
        try:
            return template.format(context=context, query=query)
        except KeyError as error:
            raise ValueError(
                f"Prompt template contains unsupported placeholder: {error.args[0]}"
            ) from error
    return build_user_content(query, context)
