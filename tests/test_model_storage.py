from base_model_handler import BaseModelHandler


def test_llama_style_tokenizer_model_is_detected(tmp_path):
    for name in ("config.json", "tokenizer.model", "model.safetensors"):
        (tmp_path / name).write_text("test", encoding="utf-8")
    handler = BaseModelHandler("example", tmp_path)

    assert handler.is_model_saved()


def test_incomplete_local_model_is_not_detected(tmp_path):
    (tmp_path / "config.json").write_text("test", encoding="utf-8")
    handler = BaseModelHandler("example", tmp_path)

    assert not handler.is_model_saved()
