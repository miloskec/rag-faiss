from pathlib import Path


def test_docker_context_excludes_historical_document_feed():
    project_root = Path(__file__).resolve().parents[1]
    ignored_entries = {
        line.strip()
        for line in (project_root / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert "workspace/feed" in ignored_entries
    assert "workspace/legacy_artifacts" in ignored_entries
