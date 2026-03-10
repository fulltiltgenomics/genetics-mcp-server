"""Integration tests for chat API endpoints."""

import json
from unittest.mock import patch


class TestStatusEndpoint:
    """Tests for /status endpoint."""

    def test_status_returns_providers(self, test_client):
        """Test that status endpoint returns available providers."""
        response = test_client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert "available_providers" in data
        assert "default_provider" in data
        assert "default_model" in data
        assert "tools_enabled" in data
        assert "available_tools" in data

    def test_status_lists_tools(self, test_client):
        """Test that status returns list of available tools."""
        response = test_client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["available_tools"], list)
        assert len(data["available_tools"]) > 0


class TestToolsEndpoint:
    """Tests for /chat/v1/tools endpoint."""

    def test_list_tools(self, test_client):
        """Test listing available tools with their definitions."""
        response = test_client.get("/chat/v1/tools")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # verify tool structure
        tool = data[0]
        assert "name" in tool
        assert "description" in tool


class TestHealthEndpoint:
    """Tests for /healthz endpoint."""

    def test_health_check(self, test_client):
        """Test health check endpoint returns ok."""
        response = test_client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_auth_info_unauthenticated(self, test_client):
        """Test /chat/v1/auth returns unauthenticated when no IAP header."""
        response = test_client.get("/chat/v1/auth")

        assert response.status_code == 200
        data = response.json()
        assert data["authenticated"] is False
        assert data["user"] is None

    def test_auth_info_authenticated(self, test_client):
        """Test /chat/v1/auth returns user from IAP header."""
        response = test_client.get(
            "/chat/v1/auth",
            headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:test@finngen.fi"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["authenticated"] is True
        assert data["user"] == "test@finngen.fi"

    def test_me_unauthenticated(self, test_client):
        """Test /chat/v1/me returns 401 when no IAP header."""
        with patch("genetics_mcp_server.auth.dependencies._require_auth", True):
            response = test_client.get("/chat/v1/me")

        assert response.status_code == 401

    def test_me_authenticated(self, test_client):
        """Test /chat/v1/me returns user from IAP header."""
        with patch("genetics_mcp_server.auth.dependencies._require_auth", True):
            response = test_client.get(
                "/chat/v1/me",
                headers={"X-Goog-Authenticated-User-Email": "accounts.google.com:test@finngen.fi"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["user"] == "test@finngen.fi"


class TestChatEndpoint:
    """Tests for /chat/v1/chat streaming endpoint."""

    def test_chat_requires_messages(self, test_client):
        """Test that chat endpoint requires messages."""
        response = test_client.post(
            "/chat/v1/chat",
            json={},
        )

        assert response.status_code == 422

    def test_chat_validates_message_format(self, test_client):
        """Test that chat validates message format."""
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [{"invalid": "format"}]
            },
        )

        assert response.status_code == 422

    def test_chat_accepts_valid_request(self, test_client):
        """Test that chat accepts valid request format."""
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "enable_tools": False,
            },
        )

        # may fail if no API key, but should not be a validation error
        assert response.status_code != 422

    def test_chat_stream_format(self, test_client):
        """Test that chat returns SSE stream format."""
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Say hello in one word"}
                ],
                "enable_tools": False,
            },
        )

        # check response headers for SSE
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type or response.status_code == 400

    def test_chat_invalid_provider(self, test_client):
        """Test error handling for invalid provider."""
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "provider": "invalid_provider",
            },
        )

        # should either reject the provider or fall back
        assert response.status_code in [200, 400, 422]


class TestChatEndpointProviders:
    """Tests for chat endpoint provider configuration."""

    def test_chat_anthropic_provider(self, test_client):
        """Test requesting Anthropic provider explicitly."""
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "provider": "anthropic",
                "enable_tools": False,
            },
        )

        # may fail if API key not set, but validates provider handling
        assert response.status_code in [200, 400]

    def test_chat_openai_provider(self, test_client):
        """Test requesting OpenAI provider explicitly."""
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "provider": "openai",
                "enable_tools": False,
            },
        )

        # may fail if API key not set
        assert response.status_code in [200, 400]

    def test_chat_custom_system_prompt(self, test_client):
        """Test providing custom system prompt."""
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "system_prompt": "You are a helpful genetics assistant.",
                "enable_tools": False,
            },
        )

        # should accept custom system prompt
        assert response.status_code in [200, 400]

    def test_chat_with_tool_profile(self, test_client):
        """Test providing tool_profile parameter."""
        for profile in ["api", "bigquery", "rag"]:
            response = test_client.post(
                "/chat/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "tool_profile": profile,
                    "enable_tools": False,
                },
            )

            # should accept tool_profile without validation error
            assert response.status_code != 422, f"tool_profile={profile} rejected"
