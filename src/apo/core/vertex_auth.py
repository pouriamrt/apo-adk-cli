"""Vertex AI authentication helpers for Google models via ADC."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


@dataclass
class VertexAIConfig:
    """Resolve Vertex AI project and location from environment variables."""

    project: str = field(default="")
    location: str = field(default="")

    def __post_init__(self) -> None:
        if not self.project:
            self.project = (
                os.environ.get("VERTEXAI_PROJECT")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
                or ""
            )
        if not self.location:
            self.location = (
                os.environ.get("VERTEXAI_LOCATION")
                or os.environ.get("GOOGLE_CLOUD_LOCATION")
                or "us-central1"
            )

    @property
    def openai_base_url(self) -> str:
        """Vertex AI OpenAI-compatible endpoint URL."""
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/{self.location}/endpoints/openapi"
        )


def get_vertex_access_token() -> str:
    """Get a bearer token from Application Default Credentials (ADC).

    Requires ``google-auth`` and valid ADC (``gcloud auth application-default login``
    or GCE/Cloud Run metadata server).
    """
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(scopes=_SCOPES)
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def is_vertex_ai_available() -> bool:
    """Check whether Vertex AI env vars are configured."""
    return bool(
        os.environ.get("VERTEXAI_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
