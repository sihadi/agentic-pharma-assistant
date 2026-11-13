"""Configuration robustness (Syntax .env quirks) and secret hygiene."""

from __future__ import annotations

import re
import subprocess

import pytest

import app.config as config
from app.config import PROJECT_ROOT, ConfigurationError, load_settings


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for name in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_BASE_UR", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_accepts_the_syntax_env_file_spelling_and_quotes(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "'k'")
    clean_env.setenv("OPENAI_BASE_UR", "'https://proxy.example'")  # as in the Syntax package
    clean_env.setenv("OPENAI_MODEL", "'chat-model'")  # docker --env-file keeps quotes
    clean_env.setenv("OPENAI_EMBEDDING_MODEL", '"emb-model"')
    s = load_settings()
    assert (s.openai_base_url, s.chat_model, s.embedding_model) == ("https://proxy.example", "chat-model", "emb-model")


def test_standard_base_url_wins_over_fallback(clean_env):
    for k, v in {"OPENAI_API_KEY": "k", "OPENAI_MODEL": "m", "OPENAI_EMBEDDING_MODEL": "e",
                 "OPENAI_BASE_URL": "https://a", "OPENAI_BASE_UR": "https://b"}.items():
        clean_env.setenv(k, v)
    assert load_settings().openai_base_url == "https://a"


def test_missing_key_is_a_clear_error(clean_env):
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        load_settings()


def test_public_view_never_exposes_the_key(settings):
    assert "test-key" not in str(settings.public_view())


def test_no_api_key_in_versionable_files():
    """Scan every file that is not git-ignored for something that looks like a key."""

    ignored_dirs = {".venv", ".git", ".merget", "assignment-pharma", "__pycache__", "cache", "traces",
                    ".pytest_cache"}
    pattern = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")
    offenders = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or ignored_dirs & set(path.relative_to(PROJECT_ROOT).parts):
            continue
        if path.name == ".env" or path.suffix in {".pdf", ".npz", ".pyc"}:
            continue
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_env_file_is_git_and_docker_ignored():
    assert re.search(r"^\.env$", (PROJECT_ROOT / ".gitignore").read_text(), re.MULTILINE)
    assert re.search(r"^\.env$", (PROJECT_ROOT / ".dockerignore").read_text(), re.MULTILINE)
    if (PROJECT_ROOT / ".git").exists():
        tracked = subprocess.run(["git", "ls-files", ".env"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert tracked.stdout.strip() == ""
