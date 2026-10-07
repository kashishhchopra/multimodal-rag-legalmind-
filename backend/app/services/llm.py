"""Pluggable multimodal LLM layer.

One interface, multiple providers (selected by LLM_PROVIDER in .env):
  * gemini  -> google-generativeai   (default; natively multimodal, free tier)
  * openai  -> OpenAI vision models   (gpt-4o / gpt-4o-mini)

The provider receives a text prompt plus a list of PIL images and returns a
plain-text answer. Secrets come only from settings/.env — never hard-coded.
If the API key is missing we raise MissingAPIKeyError so the query service can
still return the retrieved evidence (with a warning) instead of crashing.
"""
from __future__ import annotations

import base64
import io
import logging
from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from ..config import get_settings
from ..utils.errors import LLMError, MissingAPIKeyError

logger = logging.getLogger("mmrag.llm")


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[Image.Image],
        max_output_tokens: int | None = None,
    ) -> str:
        ...


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise MissingAPIKeyError("LLM_API_KEY is not set for the Gemini provider.")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        # Gemini takes the system instruction as a dedicated parameter.
        self._model = genai.GenerativeModel(model_name=model, system_instruction=None)
        self._genai = genai
        self._model_name = model
        self._system = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[Image.Image],
        max_output_tokens: int | None = None,
    ) -> str:
        try:
            model = self._genai.GenerativeModel(
                model_name=self._model_name, system_instruction=system_prompt
            )
            parts: List[object] = [user_prompt]
            parts.extend(images)  # Gemini accepts PIL images directly
            generation_config = (
                {"max_output_tokens": max_output_tokens} if max_output_tokens else None
            )
            resp = model.generate_content(parts, generation_config=generation_config)
            return (resp.text or "").strip()
        except Exception as exc:
            raise LLMError(f"Gemini generation failed: {exc}") from exc


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise MissingAPIKeyError("LLM_API_KEY is not set for the OpenAI provider.")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[Image.Image],
        max_output_tokens: int | None = None,
    ) -> str:
        try:
            content: List[dict] = [{"type": "text", "text": user_prompt}]
            for img in images:
                b64 = base64.b64encode(_pil_to_png_bytes(img)).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            kwargs: dict = {}
            if max_output_tokens:
                kwargs["max_tokens"] = max_output_tokens
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
                **kwargs,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            raise LLMError(f"OpenAI generation failed: {exc}") from exc


def get_provider() -> LLMProvider:
    """Factory: build the configured provider. Raises MissingAPIKeyError if no key."""
    s = get_settings()
    provider = s.llm_provider
    if provider == "gemini":
        return GeminiProvider(s.llm_api_key, s.llm_model)
    if provider == "openai":
        return OpenAIProvider(s.llm_api_key, s.llm_model)
    raise LLMError(f"Unknown LLM_PROVIDER: '{provider}' (expected 'gemini' or 'openai').")
