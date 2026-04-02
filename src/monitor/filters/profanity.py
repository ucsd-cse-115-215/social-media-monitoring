from better_profanity import profanity

from .base import Filter, Post


class ProfanityFilter(Filter):
    """Reject posts containing profanity (unsafe to display in class)."""

    name = "profanity"

    async def matches(self, post: Post) -> bool:
        return not profanity.contains_profanity(post.text)
