from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from google import genai

_INTEGER_LITERAL = re.compile(r'\b(?:0?[1-9]|10)\b')

SPELL_PROMPT = """Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

REWRITE_PROMPT = """Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

EXPAND_PROMPT = """Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

RERANK_PROMPT = """Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {document}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""


class LlmClient:
    __slots__ = ("model", "client")

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def enhance(self, query, *, mode=""):
        match mode:
            case "":
                return query
            case "spell":
                prompt = SPELL_PROMPT
            case "rewrite":
                prompt = REWRITE_PROMPT
            case "expand":
                prompt = EXPAND_PROMPT
            case _:
                return query
            
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt.format(query=query)
        ).text
        print(f"Enhanced query ({mode}): '{query}' -> '{response}'\n")
        return response
    
    def score(self, query: str, title: str, document: str):
        response = self.client.models.generate_content(
            model=self.model,
            contents=RERANK_PROMPT.format(
                query=query,
                title=title,
                document=document
            )
        ).text
        response_score = _INTEGER_LITERAL.findall(response)
        if response_score:
            return int(response_score[0])
        return -1
    