"""Multi-stage filtering pipeline with stats tracking."""

import time

from monitor.filters.base import Filter, Post


class Pipeline:
    """Runs a post through a sequence of filters, tracking stats at each stage.

    The key insight: filters are ordered from cheapest to most expensive.
    Each stage reduces the volume for the next, so the expensive LLM filter
    only sees a tiny fraction of the firehose.
    """

    def __init__(self, filters: list[Filter]):
        self.filters = filters
        self.stats: dict[str, dict[str, int]] = {
            f.name: {"passed": 0, "rejected": 0} for f in filters
        }
        self.total_seen = 0
        self._start_time = time.monotonic()

    async def process(self, post: Post) -> bool:
        """Run a post through all filters. Returns True if it passes all stages."""
        self.total_seen += 1

        for f in self.filters:
            passed = await f.matches(post)
            f.log(post, passed)
            if not passed:
                self.stats[f.name]["rejected"] += 1
                return False
            self.stats[f.name]["passed"] += 1

        return True

    def get_summary(self) -> dict:
        """Return pipeline stats for display."""
        elapsed = time.monotonic() - self._start_time
        return {
            "total_seen": self.total_seen,
            "elapsed_seconds": round(elapsed, 1),
            "posts_per_second": round(self.total_seen / max(elapsed, 0.1), 1),
            "stages": {
                name: {
                    "passed": s["passed"],
                    "rejected": s["rejected"],
                    "pass_rate": (
                        f"{s['passed'] / max(s['passed'] + s['rejected'], 1):.1%}"
                    ),
                }
                for name, s in self.stats.items()
            },
        }
