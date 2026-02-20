from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, base_url: str, model: str, enabled: bool, timeout_seconds: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self._disabled_due_to_error = False

    def summarize_and_tag(self, text: str) -> tuple[str, list[str], str]:
        if not self.enabled:
            return "", [], ""

        prompt = (
            "You are an assistant for offline knowledge indexing. "
            "Return strict JSON with keys: short_summary (string), bullets (array of strings), tags (array of short tags). "
            "Use Polish or English matching input language. Input:\n\n"
            f"{text[:8000]}"
        )
        raw = self._generate(prompt)
        if not raw:
            return "", [], ""

        try:
            payload = json.loads(self._extract_json(raw))
            summary = payload.get("short_summary", "")
            bullets = payload.get("bullets", [])
            tags = payload.get("tags", [])
            bullet_text = "\n".join(f"- {b}" for b in bullets if isinstance(b, str))
            full_summary = f"{summary}\n{bullet_text}".strip()
            clean_tags = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
            language = self._guess_language(text)
            return full_summary, clean_tags, language
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM JSON parsing failed: %s", exc)
            return "", [], self._guess_language(text)

    def answer(self, query: str, context: str) -> str:
        if not self.enabled:
            if context.strip():
                return "Local LLM unavailable. Returning retrieval context only.\n\n" + context[:2000]
            return "Local LLM unavailable and no indexed context matched the query."
        prompt = (
            "You are an offline assistant for the user's own authorized internal training notes. "
            "The context below comes from files the user indexed locally. "
            "Answer using only the provided context. Do not refuse unless context is empty. "
            "If context is empty, say you do not have enough indexed data yet. "
            "Answer in the language of the question (Polish or English).\n\n"
            f"Question: {query}\n\nContext:\n{context[:12000]}"
        )
        response = self._generate(prompt)
        if response and not self._looks_like_refusal(response):
            return response
        if context.strip():
            return self._fallback_from_context(query, context)
        return "No model response and no indexed context matched the query."

    def _generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=(3, self.timeout_seconds),
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM request failed: %s", exc)
            if not self._disabled_due_to_error:
                logger.warning("Disabling local LLM after request failure. Set llm.enabled=true once Ollama is available.")
                self._disabled_due_to_error = True
            self.enabled = False
            return ""

    @staticmethod
    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return "{}"
        return text[start : end + 1]

    @staticmethod
    def _guess_language(text: str) -> str:
        polish_chars = {"ą", "ć", "ę", "ł", "ń", "ó", "ś", "ź", "ż"}
        return "pl" if any(ch in text.lower() for ch in polish_chars) else "en"

    @staticmethod
    def _looks_like_refusal(text: str) -> bool:
        low = text.lower()
        refusal_markers = [
            "nie mogę",
            "nie moge",
            "i can't",
            "i cannot",
            "can't help with",
            "nie mogę udostępnić",
            "nie moge udostepnic",
            "illegal",
            "szkodliwe",
        ]
        return any(marker in low for marker in refusal_markers)

    @staticmethod
    def _fallback_from_context(query: str, context: str) -> str:
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        top = lines[:10]
        if not top:
            return "No indexed context matched the query."
        return (
            "Model returned a refusal or empty answer. Returning context-based summary fallback.\n\n"
            f"Query: {query}\n"
            + "\n".join(f"- {line}" for line in top)
        )
