from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DiarizationService:
    def __init__(self, model_path: str | None, enabled: bool, gpu_enabled: bool = False) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self.gpu_enabled = gpu_enabled
        self.pipeline = None

        if not enabled:
            return
        if not model_path:
            logger.warning("Diarization enabled but model_path is empty. Disabling diarization.")
            self.enabled = False
            return

        try:
            from pyannote.audio import Pipeline

            local_model = Path(model_path)
            if local_model.exists():
                self.pipeline = Pipeline.from_pretrained(str(local_model))
            else:
                self.pipeline = Pipeline.from_pretrained(model_path)
            if gpu_enabled:
                import torch

                self.pipeline.to(torch.device("cuda"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Diarization init failed, disabling: %s", exc)
            self.enabled = False

    def diarize(self, audio_path: Path) -> list[dict]:
        if not self.enabled or self.pipeline is None:
            return []
        annotation = self.pipeline(str(audio_path))
        speakers: dict[str, str] = {}
        mapped: list[dict] = []
        next_idx = 1
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers[speaker] = f"Speaker {next_idx}"
                next_idx += 1
            mapped.append(
                {
                    "speaker": speakers[speaker],
                    "start": float(segment.start),
                    "end": float(segment.end),
                }
            )
        return mapped


def align_speakers(transcript_segments: list[dict], speaker_segments: list[dict]) -> list[dict]:
    if not speaker_segments:
        return transcript_segments

    for seg in transcript_segments:
        ts = float(seg.get("start", 0.0))
        seg["speaker"] = _speaker_for_time(ts, speaker_segments)
    return transcript_segments


def _speaker_for_time(ts: float, speaker_segments: list[dict]) -> str:
    for s in speaker_segments:
        if float(s["start"]) <= ts <= float(s["end"]):
            return str(s["speaker"])
    return "Speaker 1"
