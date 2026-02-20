from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class VisionService:
    def __init__(
        self,
        enabled: bool,
        base_url: str,
        model: str,
        max_frames: int,
        timeout_seconds: int,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_frames = max_frames
        self.timeout_seconds = timeout_seconds

    def describe_frames(self, frame_paths: list[Path]) -> list[str]:
        if not self.enabled or not frame_paths:
            return []

        selected = frame_paths[: self.max_frames]
        descriptions: list[str] = []
        for idx, frame_path in enumerate(selected, start=1):
            text = self._describe_frame(frame_path)
            if text:
                descriptions.append(f"Frame {idx}: {text}")
        return descriptions

    def _describe_frame(self, frame_path: Path) -> str:
        try:
            image_b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
            prompt = (
                "You are analyzing a technical training slide screenshot. "
                "Extract only grounded visual information: key entities, relationships, chart trends, "
                "and diagram flow. Do not guess unknown details. Keep answer concise."
            )
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=(5, self.timeout_seconds),
            )
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision description failed for %s: %s", frame_path, exc)
            return ""
