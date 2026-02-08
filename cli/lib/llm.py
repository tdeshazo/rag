from __future__ import annotations

from google import genai


SPELL_PROMPT = """Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""


class LlmClient:
    __slots__ = ("model", "client")

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def spell(self, query):
        return self.client.models.generate_content(
            model=self.model,
            contents=SPELL_PROMPT.format(query=query)
        )
