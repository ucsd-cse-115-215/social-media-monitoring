from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging and stats."""
        ...

    @abstractmethod
    async def matches(self, post: Post) -> bool:
        """Return True if the post should advance to the next stage."""
        ...
