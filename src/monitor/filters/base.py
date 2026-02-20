from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json


@dataclass
class Post:
    """A Bluesky post extracted from the firehose."""

    text: str
    author_did: str
    created_at: datetime
    langs: list[str] | None = None
    url: str | None = None
    # Filters can attach metadata here (e.g. LLM analysis)
    metadata: dict[str, Any] = field(default_factory=dict)


class Filter(ABC):
    """Base class for a pipeline filter.

    Each filter is a stage in the pipeline. It inspects a Post and decides
    whether to keep it (True) or drop it (False). Filters can also annotate
    the post by writing to post.metadata.

    Set log_path to a file path to enable JSONL logging for this filter.
    """

    log_path: Path | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging and stats."""
        ...

    @abstractmethod
    async def matches(self, post: Post) -> bool:
        """Return True if the post should advance to the next stage."""
        ...

    def log(self, post: Post, passed: bool) -> None:
        """Write a JSONL entry for this filter decision. No-op if log_path is None."""
        if self.log_path is None:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filter": self.name,
            "passed": passed,
            "text": post.text,
            "url": post.url,
        }
        if post.metadata:
            entry["metadata"] = post.metadata
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
