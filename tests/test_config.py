from pathlib import Path

import rag_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_relative_log_directory_resolves_from_project_root():
    resolved = rag_config.resolve_project_path(
        "workspace/logs", "/unused/default"
    )

    assert resolved == (PROJECT_ROOT / "workspace" / "logs").resolve()


def test_absolute_log_directory_is_preserved(tmp_path):
    resolved = rag_config.resolve_project_path(str(tmp_path), "/unused/default")

    assert resolved == tmp_path.resolve()


def test_hf_token_placeholder_is_ignored(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "<YOUR_HF_TOKEN>")

    assert rag_config.get_hf_token() is None


def test_hf_token_placeholder_is_removed_from_process_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "<YOUR_HF_TOKEN>")

    rag_config.discard_placeholder_hf_token()

    assert "HF_TOKEN" not in rag_config.os.environ


def test_hf_token_uses_canonical_environment_name(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    assert rag_config.get_hf_token() == "test-token"


def test_project_uses_only_canonical_hf_token_name():
    files = [
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / "docker-compose.yaml",
        PROJECT_ROOT / "Readme.md",
        PROJECT_ROOT / "workspace" / "base_model_handler.py",
        PROJECT_ROOT / "workspace" / "rag_config.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    assert "HF_TOKEN" in combined
    assert "HUGGINGFACE_TOKEN" not in combined
    assert "HUGGING_FACE_HUB_TOKEN" not in combined


def test_hugging_face_config_loads_before_transformers_import():
    source = (PROJECT_ROOT / "workspace" / "base_model_handler.py").read_text(
        encoding="utf-8"
    )

    assert source.index("from rag_config") < source.index("from transformers")


def test_default_model_is_prequantized_public_qwen():
    source = (PROJECT_ROOT / "workspace" / "rag_config.py").read_text(
        encoding="utf-8"
    )

    assert '"MODEL_ID", "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"' in source
