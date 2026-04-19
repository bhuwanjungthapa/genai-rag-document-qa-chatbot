"""
LLM provider abstraction.

Supports:
- Gemini (`google-generativeai`) - default
- OpenAI (`openai>=1.0`) - optional fallback
- A `NullClient` used when no API key is configured: returns a safe message so
  that ingestion/retrieval still work without a live LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    name: str = "null"

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's answer as plain text."""


class NullClient(LLMClient):
    name = "none"

    def generate(self, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        return (
            "Answer generation is disabled because no LLM API key is configured.\n"
            "Retrieval still works; add GEMINI_API_KEY or OPENAI_API_KEY in `.env` "
            "to enable grounded answer generation."
        )


class GeminiClient(LLMClient):
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        import google.generativeai as genai

        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        model = self._genai.GenerativeModel(
            self.model_name,
            system_instruction=system_prompt,
        )
        try:
            resp = model.generate_content(
                user_prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 1024,
                },
            )
            return (resp.text or "").strip()
        except Exception as e:  # noqa: BLE001
            return f"[Gemini error] {e}"


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty")
        self._client = OpenAI(api_key=api_key)
        self.model_name = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            return f"[OpenAI error] {e}"


def build_llm_client(
    provider: str,
    *,
    gemini_api_key: str = "",
    gemini_model: str = "gemini-1.5-flash",
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
) -> LLMClient:
    """Factory that picks the right client based on the requested provider."""
    provider = (provider or "").lower().strip()

    if provider == "gemini" and gemini_api_key:
        return GeminiClient(api_key=gemini_api_key, model=gemini_model)
    if provider == "openai" and openai_api_key:
        return OpenAIClient(api_key=openai_api_key, model=openai_model)

    # Silent fallbacks: if the requested provider isn't available but the other is.
    if gemini_api_key:
        return GeminiClient(api_key=gemini_api_key, model=gemini_model)
    if openai_api_key:
        return OpenAIClient(api_key=openai_api_key, model=openai_model)

    return NullClient()


def describe_available(
    gemini_api_key: str, openai_api_key: str
) -> Optional[str]:
    """Human-readable summary of which LLM providers are ready."""
    parts = []
    if gemini_api_key:
        parts.append("Gemini")
    if openai_api_key:
        parts.append("OpenAI")
    if not parts:
        return None
    return " + ".join(parts)
