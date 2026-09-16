from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas.research import ResearchResponse

SYSTEM_PROMPT = """
You are an AI business research assistant.

Your job is to analyze the user's research question
and produce a concise, structured response.

Important rules:

1. Do not invent facts or sources.
2. Clearly distinguish assumptions from known information.
3. Focus on useful business implications.
4. If information is insufficient, say so.
5. Return the requested structured format.
"""


class ResearchService:
    def __init__(self) -> None:
        settings = get_settings()

        client_kwargs = {
            "api_key": settings.openai_api_key,
        }

        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = settings.openai_model

    async def research(self, question: str) -> ResearchResponse:

        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            response_format=ResearchResponse,
        )

        result = response.choices[0].message.parsed

        if result is None:
            raise RuntimeError("LLM returned no structured response")

        return result
