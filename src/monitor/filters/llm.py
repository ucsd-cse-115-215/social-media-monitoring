import json
import logging

import openai

from .base import Filter, Post

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a poetry detector. Given a social media post, determine whether it
contains an original poem (not just a quote or song lyrics).

Respond with JSON:
{
  "is_poetry": true/false,
  "confidence": 0.0-1.0,
  "explanation": "brief reason"
}
"""


class LLMFilter(Filter):
    """Stage 3: LLM-based poetry classification.

    This is the expensive stage — it makes an API call for every post it sees.
    That's why it must be last in the pipeline, so the cheaper filters have
    already eliminated most of the firehose.
    """

    name = "llm"

    def __init__(
        self,
        client: openai.AsyncOpenAI,
        model: str = "gpt-5.4",
        confidence_threshold: float = 0.5,
        name: str = "llm",
    ):
        self.client = client
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.name = name

    async def matches(self, post: Post) -> bool:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": post.text},
                ],
                response_format={"type": "json_object"},
                # max_completion_tokens=150,
            )
            result = json.loads(response.choices[0].message.content)

            # Attach the LLM's analysis and token usage to the post
            post.metadata["llm_analysis"] = result
            if response.usage:
                post.metadata["usage"] = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }

            return (
                result.get("is_poetry", False)
                and result.get("confidence", 0) >= self.confidence_threshold
            )

        except Exception:
            logger.exception("LLM filter error")
            return False
