from .base import Filter, Post


class BasicFilter(Filter):
    """Stage 1: Cheap checks to discard obviously irrelevant posts.

    This runs on every single post from the firehose, so it must be fast.
    No network calls, no heavy computation — just simple field checks.
    """

    name = "basic"

    def __init__(self, min_length: int = 20, required_langs: set[str] = {"en"}):
        self.min_length = min_length
        self.required_langs = required_langs

    async def matches(self, post: Post) -> bool:
        # Must have non-empty text
        if not post.text or not post.text.strip():
            return False

        # Must meet minimum length (very short posts are unlikely to be poems)
        if len(post.text.strip()) < self.min_length:
            return False

        # Language filter: if the post declares languages, at least one must match
        if post.langs and not (set(post.langs) & self.required_langs):
            return False

        return True
