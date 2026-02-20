from .base import Filter, Post


class StructuralFilter(Filter):
    """Stage 2: Structural heuristics for poetry-like text.

    Poems tend to have a distinctive visual shape: multiple short lines,
    deliberate line breaks, and they don't look like bullet-point lists.
    This filter checks for that shape — cheaply, without understanding content.
    """

    name = "structural"

    def __init__(self, min_lines: int = 3, max_avg_line_length: int = 60):
        self.min_lines = min_lines
        self.max_avg_line_length = max_avg_line_length

    async def matches(self, post: Post) -> bool:
        lines = [line.strip() for line in post.text.strip().split("\n") if line.strip()]

        # Poems have multiple lines
        if len(lines) < self.min_lines:
            return False

        # Lines should be relatively short (not long paragraphs)
        avg_length = sum(len(line) for line in lines) / len(lines)
        if avg_length > self.max_avg_line_length:
            return False

        # Filter out lists (bullet points, numbered items)
        list_markers = sum(
            1
            for line in lines
            if line[0] in "-*•" or (len(line) > 1 and line[0].isdigit() and line[1] in ".)")
        )
        if list_markers > len(lines) // 2:
            return False

        return True
