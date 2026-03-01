"""Tests for Vertex AI auth helpers and CLI model detection."""

from unittest.mock import MagicMock, patch

import pytest

from apo.core.vertex_auth import VertexAIConfig, get_vertex_access_token, is_vertex_ai_available
from apo.cli.commands import _detect_default_model, _resolve_vertex_mode


# ---------------------------------------------------------------------------
# VertexAIConfig
# ---------------------------------------------------------------------------

class TestVertexAIConfig:
    def test_resolves_from_vertexai_env(self, monkeypatch):
        monkeypatch.setenv("VERTEXAI_PROJECT", "proj-a")
        monkeypatch.setenv("VERTEXAI_LOCATION", "europe-west1")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

        cfg = VertexAIConfig()
        assert cfg.project == "proj-a"
        assert cfg.location == "europe-west1"

    def test_resolves_from_google_cloud_env(self, monkeypatch):
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
        monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-b")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-east1")

        cfg = VertexAIConfig()
        assert cfg.project == "proj-b"
        assert cfg.location == "asia-east1"

    def test_vertexai_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("VERTEXAI_PROJECT", "proj-a")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-b")

        cfg = VertexAIConfig()
        assert cfg.project == "proj-a"

    def test_default_location(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-c")
        monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

        cfg = VertexAIConfig()
        assert cfg.location == "us-central1"

    def test_openai_base_url(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
        monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

        cfg = VertexAIConfig()
        assert cfg.openai_base_url == (
            "https://us-central1-aiplatform.googleapis.com/v1/"
            "projects/my-proj/locations/us-central1/endpoints/openapi"
        )

    def test_explicit_args_override_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
        cfg = VertexAIConfig(project="arg-proj", location="arg-loc")
        assert cfg.project == "arg-proj"
        assert cfg.location == "arg-loc"


# ---------------------------------------------------------------------------
# get_vertex_access_token
# ---------------------------------------------------------------------------

class TestGetVertexAccessToken:
    @patch("google.auth.transport.requests.Request")
    @patch("google.auth.default")
    def test_returns_token(self, mock_default, mock_request):
        mock_creds = MagicMock()
        mock_creds.token = "fake-token-123"
        mock_default.return_value = (mock_creds, "project-id")

        token = get_vertex_access_token()

        assert token == "fake-token-123"
        mock_creds.refresh.assert_called_once()


# ---------------------------------------------------------------------------
# is_vertex_ai_available
# ---------------------------------------------------------------------------

class TestIsVertexAIAvailable:
    def test_true_with_vertexai_project(self, monkeypatch):
        monkeypatch.setenv("VERTEXAI_PROJECT", "proj")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        assert is_vertex_ai_available() is True

    def test_true_with_google_cloud_project(self, monkeypatch):
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
        assert is_vertex_ai_available() is True

    def test_false_when_no_project(self, monkeypatch):
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        assert is_vertex_ai_available() is False


# ---------------------------------------------------------------------------
# _detect_default_model
# ---------------------------------------------------------------------------

class TestDetectDefaultModel:
    def test_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        model, use_vertex = _detect_default_model()
        assert model == "openai/gpt-5.2"
        assert use_vertex is False

    def test_google_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-xxx")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        model, use_vertex = _detect_default_model()
        assert model == "gemini/gemini-2.5-flash"
        assert use_vertex is False

    def test_anthropic_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")

        model, use_vertex = _detect_default_model()
        assert model == "anthropic/claude-sonnet-4-6"
        assert use_vertex is False

    def test_vertex_auto_detect(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")

        model, use_vertex = _detect_default_model()
        assert model == "vertex_ai/gemini-2.5-flash"
        assert use_vertex is True

    def test_no_credentials_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)

        model, use_vertex = _detect_default_model()
        assert model == "gemini/gemini-2.5-flash"
        assert use_vertex is False


# ---------------------------------------------------------------------------
# _resolve_vertex_mode
# ---------------------------------------------------------------------------

class TestResolveVertexMode:
    def test_vertex_prefix_auto_enables(self):
        model, use_vertex = _resolve_vertex_mode("vertex_ai/gemini-2.5-pro", False)
        assert model == "vertex_ai/gemini-2.5-pro"
        assert use_vertex is True

    def test_flag_rewrites_gemini_prefix(self):
        model, use_vertex = _resolve_vertex_mode("gemini/gemini-2.5-flash", True)
        assert model == "vertex_ai/gemini-2.5-flash"
        assert use_vertex is True

    def test_flag_with_non_gemini_model(self):
        model, use_vertex = _resolve_vertex_mode("openai/gpt-4o", True)
        assert model == "openai/gpt-4o"
        assert use_vertex is True

    def test_no_flag_no_prefix(self):
        model, use_vertex = _resolve_vertex_mode("gemini/gemini-2.5-flash", False)
        assert model == "gemini/gemini-2.5-flash"
        assert use_vertex is False
