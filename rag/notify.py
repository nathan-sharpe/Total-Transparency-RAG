"""Error-webhook notification: the outbound half of the backstop guardrail.

The global exception handler (main.py) calls notify_error() when
ERROR_WEBHOOK_URL is set. The payload carries both a "text" field (Slack
incoming webhooks) and a "content" field (Discord webhooks) with the same
message — each service ignores the other's field, so one URL setting covers
both without a provider switch.

A notification failure must never break the API response it accompanies:
every failure path here is caught and logged, nothing propagates.
"""

import logging

import httpx

from rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


async def notify_error(
    message: str,
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> bool:
    """POST the message to the configured webhook. Returns True if delivered.

    No-op returning False when ERROR_WEBHOOK_URL is unset. The transport
    parameter exists so unit tests can inject httpx.MockTransport — the same
    pattern as OllamaEmbedder.
    """
    settings = settings or get_settings()
    url = settings.error_webhook_url
    if not url:
        return False
    try:
        async with httpx.AsyncClient(
            timeout=settings.webhook_timeout_seconds, transport=transport
        ) as client:
            response = await client.post(
                url.get_secret_value(), json={"text": message, "content": message}
            )
            response.raise_for_status()
        logger.info("error-webhook notification delivered")
        return True
    except Exception:
        logger.warning("error-webhook notification failed", exc_info=True)
        return False
