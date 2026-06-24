"""Integration tests for chat API endpoints."""

import json
from unittest.mock import patch

from genetics_mcp_server.llm_service import StreamChunk


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

    def test_chat_rejects_too_long_message(self, test_client):
        """Typed text over the limit is rejected with 413."""
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [{"role": "user", "content": "x" * 50001}],
                "enable_tools": False,
            },
        )

        assert response.status_code == 413

    def test_chat_excludes_attachments_from_length(self, test_client):
        """A large data-file attachment block does not count toward the text limit."""
        big_file_block = {"type": "text", "text": "[File: data.tsv]\n" + ("a\tb\n" * 100000)}
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [{"role": "user", "content": [big_file_block, {"type": "text", "text": "analyze"}]}],
                "enable_tools": False,
            },
        )

        # not a 413 (size) error; may be 200/400 depending on provider availability
        assert response.status_code != 413

    def test_chat_rejects_too_many_attachments(self, test_client):
        """More than the allowed attachment blocks per message is rejected with 413."""
        blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}} for _ in range(11)]
        response = test_client.post(
            "/chat/v1/chat",
            json={
                "messages": [{"role": "user", "content": blocks}],
                "enable_tools": False,
            },
        )

        assert response.status_code == 413

    def test_chat_stream_emits_usage_event(self, test_client):
        """Test that a usage StreamChunk is emitted as an SSE event with usage data."""
        usage_payload = {
            "iteration": 1,
            "input_tokens": 1500,
            "output_tokens": 200,
            "total_input_tokens": 1500,
            "total_output_tokens": 200,
            "context_window": 200000,
            "context_percent": 0.8,
        }

        async def mock_stream(**kwargs):
            yield StreamChunk(type="usage", content=json.dumps(usage_payload))
            yield StreamChunk(
                type="done",
                content="",
                message_content=[{"type": "text", "text": "Hello!"}],
            )

        with patch(
            "genetics_mcp_server.chat_api.get_llm_service"
        ) as mock_get_service:
            mock_service = mock_get_service.return_value
            mock_service.anthropic_client = True
            mock_service.openai_client = None
            mock_service.stream_chat = mock_stream

            response = test_client.post(
                "/chat/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "provider": "anthropic",
                    "enable_tools": False,
                },
            )

        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type

        # parse SSE events from response body
        events = []
        for line in response.text.splitlines():
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str:
                    events.append(json.loads(data_str))

        # find the usage event
        usage_events = [e for e in events if e.get("type") == "usage"]
        assert len(usage_events) == 1
        usage_event = usage_events[0]
        assert usage_event["iteration"] == 1
        assert usage_event["input_tokens"] == 1500
        assert usage_event["output_tokens"] == 200
        assert usage_event["total_input_tokens"] == 1500
        assert usage_event["total_output_tokens"] == 200
        assert usage_event["context_window"] == 200000
        assert usage_event["context_percent"] == 0.8

        # verify done event also present
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1


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
