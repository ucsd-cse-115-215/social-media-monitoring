"""Bluesky Jetstream firehose client.

Jetstream is a public WebSocket service that streams all Bluesky activity.
No authentication required — just connect and receive JSON events.
Docs: https://github.com/bluesky-social/jetstream
"""

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

import websockets

from monitor.filters.base import Post

logger = logging.getLogger(__name__)

JETSTREAM_URL = "wss://jetstream2.us-east.bsky.network/subscribe"


async def stream_posts(
    collections: list[str] | None = None,
) -> AsyncIterator[Post]:
    """Connect to Jetstream and yield Post objects for new Bluesky posts.

    Automatically reconnects on disconnection.
    """
    params = collections or ["app.bsky.feed.post"]
    query = "&".join(f"wantedCollections={c}" for c in params)
    url = f"{JETSTREAM_URL}?{query}"

    while True:
        try:
            async for ws in websockets.connect(url):
                logger.info("Connected to Jetstream")
                try:
                    async for raw in ws:
                        post = _parse_event(raw)
                        if post is not None:
                            yield post
                except websockets.ConnectionClosed:
                    logger.warning("Jetstream connection closed, reconnecting...")
                    continue
        except Exception:
            logger.exception("Jetstream connection error, reconnecting...")


def _parse_event(raw: str) -> Post | None:
    """Parse a Jetstream event into a Post, or None if not a new post."""
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if event.get("kind") != "commit":
        return None

    commit = event.get("commit", {})
    if commit.get("operation") != "create":
        return None
    if commit.get("collection") != "app.bsky.feed.post":
        return None

    record = commit.get("record", {})
    text = record.get("text", "")
    if not text:
        return None

    did = event.get("did", "")
    rkey = commit.get("rkey", "")
    created_str = record.get("createdAt", "")

    try:
        created_at = datetime.fromisoformat(created_str)
    except (ValueError, TypeError):
        created_at = datetime.now(timezone.utc)

    return Post(
        text=text,
        author_did=did,
        created_at=created_at,
        langs=record.get("langs"),
        url=f"https://bsky.app/profile/{did}/post/{rkey}" if did and rkey else None,
    )
