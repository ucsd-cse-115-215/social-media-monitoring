"""Tests for the filtering pipeline.

Run with: python -m pytest tests/
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from monitor.filters.base import Post
from monitor.filters.basic import BasicFilter
from monitor.filters.structural import StructuralFilter

SAMPLE_POSTS_PATH = Path(__file__).parent / "sample_posts.json"


def _make_post(**overrides) -> Post:
    """Helper to create a Post with sensible defaults."""
    defaults = {
        "text": "hello world",
        "author_did": "did:plc:test",
        "created_at": datetime.now(timezone.utc),
        "langs": ["en"],
    }
    defaults.update(overrides)
    return Post(**defaults)


def load_sample_posts() -> list[dict]:
    with open(SAMPLE_POSTS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# BasicFilter tests
# ---------------------------------------------------------------------------

class TestBasicFilter:
    @pytest.fixture
    def f(self):
        return BasicFilter(min_length=20)

    @pytest.mark.asyncio
    async def test_rejects_empty(self, f):
        assert not await f.matches(_make_post(text=""))

    @pytest.mark.asyncio
    async def test_rejects_short(self, f):
        assert not await f.matches(_make_post(text="hi"))

    @pytest.mark.asyncio
    async def test_rejects_wrong_language(self, f):
        assert not await f.matches(
            _make_post(text="a" * 30, langs=["es"])
        )

    @pytest.mark.asyncio
    async def test_accepts_english_long_enough(self, f):
        assert await f.matches(
            _make_post(text="a" * 30, langs=["en"])
        )

    @pytest.mark.asyncio
    async def test_accepts_no_language_tag(self, f):
        """Posts without language tags should pass (benefit of the doubt)."""
        assert await f.matches(_make_post(text="a" * 30, langs=None))


# ---------------------------------------------------------------------------
# StructuralFilter tests
# ---------------------------------------------------------------------------

class TestStructuralFilter:
    @pytest.fixture
    def f(self):
        return StructuralFilter(min_lines=3)

    @pytest.mark.asyncio
    async def test_rejects_single_line(self, f):
        assert not await f.matches(_make_post(text="Just a regular tweet, nothing special here."))

    @pytest.mark.asyncio
    async def test_accepts_multi_line_short_lines(self, f):
        poem = "roses are red\nviolets are blue\nsugar is sweet\nand so are you"
        assert await f.matches(_make_post(text=poem))

    @pytest.mark.asyncio
    async def test_rejects_bullet_list(self, f):
        text = "- buy milk\n- buy eggs\n- buy bread\n- buy butter"
        assert not await f.matches(_make_post(text=text))

    @pytest.mark.asyncio
    async def test_rejects_numbered_list(self, f):
        text = "1. First thing\n2. Second thing\n3. Third thing\n4. Fourth thing"
        assert not await f.matches(_make_post(text=text))

    @pytest.mark.asyncio
    async def test_rejects_long_paragraphs(self, f):
        text = "\n".join(["x" * 80] * 4)
        assert not await f.matches(_make_post(text=text))


# ---------------------------------------------------------------------------
# Integration: sample posts through basic + structural
# ---------------------------------------------------------------------------

class TestSamplePosts:
    """Smoke test: run sample posts through the cheap filters."""

    @pytest.mark.asyncio
    async def test_sample_posts_basic_filter(self):
        f = BasicFilter(min_length=20)
        posts = load_sample_posts()

        for p in posts:
            post = _make_post(text=p["text"], langs=p.get("langs"))
            result = await f.matches(post)
            # "ok" and the Spanish post should be rejected
            if p["text"] == "ok" or p.get("langs") == ["es"]:
                assert not result, f"Expected rejection: {p['text'][:40]}"

    @pytest.mark.asyncio
    async def test_poetry_passes_structural(self):
        f = StructuralFilter(min_lines=3)
        posts = load_sample_posts()

        for p in posts:
            if p["label"] == "poetry":
                post = _make_post(text=p["text"])
                assert await f.matches(post), (
                    f"Poetry should pass structural filter: {p['text'][:40]}"
                )
