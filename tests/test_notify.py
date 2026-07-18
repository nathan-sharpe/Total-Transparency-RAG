"""Webhook notifier tests: delivery, the disabled path, and the rule that a
notification failure never propagates (it must not break the API response it
accompanies)."""

import asyncio
import json

import httpx

from rag.config import Settings
from rag.notify import notify_error


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, postgres_user="t", postgres_password="t", **overrides)


def test_unset_url_means_no_request_and_not_delivered():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200)

    delivered = asyncio.run(
        notify_error("boom", settings=make_settings(), transport=httpx.MockTransport(handler))
    )
    assert delivered is False
    assert calls == []


def test_delivers_slack_and_discord_compatible_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200)

    settings = make_settings(error_webhook_url="https://hooks.example/T00/B00")
    delivered = asyncio.run(
        notify_error(
            "[rag-api] error_id=abc", settings=settings, transport=httpx.MockTransport(handler)
        )
    )
    assert delivered is True
    assert captured["url"] == "https://hooks.example/T00/B00"
    # Slack reads "text", Discord reads "content"; both carry the message.
    assert captured["json"]["text"] == "[rag-api] error_id=abc"
    assert captured["json"]["content"] == "[rag-api] error_id=abc"


def test_http_error_status_is_swallowed():
    settings = make_settings(error_webhook_url="https://hooks.example/x")
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    assert asyncio.run(notify_error("boom", settings=settings, transport=transport)) is False


def test_connection_failure_is_swallowed():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    settings = make_settings(error_webhook_url="https://hooks.example/x")
    transport = httpx.MockTransport(handler)
    assert asyncio.run(notify_error("boom", settings=settings, transport=transport)) is False
