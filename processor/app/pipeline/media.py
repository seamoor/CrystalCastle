from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from app.pipeline.diarization import DiarizationService, align_speakers
from app.pipeline.vision import VisionService

logger = logging.getLogger(__name__)


class MediaProcessor:
    def __init__(
        self,
        data_dir: Path,
        whisper_model_size: str,
        whisper_device: str,
        whisper_compute_type: str,
        ocr_enabled: bool,
        ocr_fps: float,
        slide_change_threshold: int,
        ocr_languages: list[str],
        diarization_service: DiarizationService,
        vision_service: VisionService,
    ) -> None:
        self.data_dir = data_dir
        self.whisper_model_size = whisper_model_size
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.ocr_enabled = ocr_enabled
        self.ocr_fps = ocr_fps
        self.slide_change_threshold = slide_change_threshold
        self.ocr_languages = ocr_languages
        self.diarization_service = diarization_service
        self.vision_service = vision_service
        self._ocr_unavailable_reason: str | None = None

        if self.ocr_enabled:
            try:
                import paddle  # noqa: F401
                logger.info("Paddle runtime detected. Slide OCR is enabled.")
            except Exception as exc:  # noqa: BLE001
                self.ocr_enabled = False
                self._ocr_unavailable_reason = str(exc)
                logger.warning(
                    "Disabling OCR because paddle runtime is unavailable: %s. "
                    "Install paddlepaddle in the processor image or set ocr.enabled=false.",
                    exc,
                )

    def process(
        self,
        path: Path,
        progress_cb: Callable[[str, float, dict[str, Any] | None], None] | None = None,
    ) -> dict:
        work_dir = self.data_dir / "jobs" / path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Media processing started: path=%s work_dir=%s", path, work_dir)

        audio_path = work_dir / "audio.wav"
        logger.info("Extracting audio track: input=%s output=%s", path, audio_path)
        self._extract_audio(path, audio_path)
        self._report(progress_cb, "extract_audio", 1.0, {"audio_path": str(audio_path)})

        audio_duration = _probe_duration_seconds(audio_path)

        logger.info("Transcription started: audio=%s", audio_path)
        transcript = self._transcribe(audio_path, audio_duration=audio_duration, progress_cb=progress_cb)
        segments = transcript["segments"]
        logger.info(
            "Transcription finished: language=%s segments=%d",
            transcript.get("language"),
            len(segments),
        )

        logger.info("Diarization started: audio=%s enabled=%s", audio_path, self.diarization_service.enabled)
        speakers = self.diarization_service.diarize(audio_path)
        segments = align_speakers(segments, speakers)
        logger.info("Diarization finished: speaker_segments=%d", len(speakers))

        logger.info("Slide OCR started: media=%s enabled=%s", path, self.ocr_enabled)
        slide_text, vision_text = self._extract_slides_text(path, work_dir, progress_cb=progress_cb)
        logger.info("Slide OCR finished: extracted_chars=%d", len(slide_text))
        transcript_text = "\n".join(
            self._segment_to_line(seg) for seg in segments
        )

        merged_text = transcript_text
        if slide_text:
            merged_text += "\n\n[SLIDES OCR]\n" + slide_text
        if vision_text:
            merged_text += "\n\n[SLIDES VISION]\n" + vision_text

        duration = max((float(seg.get("end", 0.0)) for seg in segments), default=0.0)
        speaker_names = sorted({s.get("speaker", "Speaker 1") for s in segments if s.get("speaker")})

        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info(
            "Media processing finished: path=%s duration_seconds=%.2f speakers=%d",
            path,
            duration,
            len(speaker_names),
        )
        return {
            "text": merged_text,
            "language": transcript.get("language"),
            "duration_seconds": duration,
            "segments": segments,
            "speakers": speaker_names,
        }

    def _extract_audio(self, input_path: Path, output_audio_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_audio_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _transcribe(
        self,
        audio_path: Path,
        audio_duration: float,
        progress_cb: Callable[[str, float, dict[str, Any] | None], None] | None = None,
    ) -> dict:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("faster-whisper is unavailable") from exc

        device = self.whisper_device
        if device == "auto":
            device = "cuda" if _has_cuda() else "cpu"

        model = WhisperModel(
            self.whisper_model_size,
            device=device,
            compute_type=self.whisper_compute_type,
        )
        logger.info("Transcription heartbeat started: audio=%s interval_seconds=30", audio_path)
        self._report(progress_cb, "transcription", 0.0, {"audio_duration_seconds": audio_duration})
        segments, info = model.transcribe(str(audio_path), vad_filter=True, word_timestamps=False)

        out_segments: list[dict] = []
        last_beat = time.monotonic()
        for seg in segments:
            out_segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                }
            )
            now = time.monotonic()
            progress = 0.0
            if audio_duration > 0:
                progress = min(1.0, float(seg.end) / audio_duration)
            if now - last_beat >= 30:
                logger.info(
                    "Transcription heartbeat: audio=%s collected_segments=%d last_end=%.2f stage_progress=%.1f%%",
                    audio_path,
                    len(out_segments),
                    float(seg.end),
                    progress * 100.0,
                )
                self._report(
                    progress_cb,
                    "transcription",
                    progress,
                    {"collected_segments": len(out_segments), "last_end_seconds": float(seg.end)},
                )
                last_beat = now

        self._report(progress_cb, "transcription", 1.0, {"collected_segments": len(out_segments)})

        return {
            "language": getattr(info, "language", None),
            "segments": out_segments,
        }

    def _extract_slides_text(
        self,
        media_path: Path,
        work_dir: Path,
        progress_cb: Callable[[str, float, dict[str, Any] | None], None] | None = None,
    ) -> tuple[str, str]:
        if media_path.suffix.lower() in {".mp3", ".wav", ".m4a"}:
            self._report(progress_cb, "ocr_sampling", 1.0, {"skipped": True, "reason": "audio_only"})
            self._report(progress_cb, "ocr_inference", 1.0, {"skipped": True, "reason": "audio_only"})
            self._report(progress_cb, "vision_inference", 1.0, {"skipped": True, "reason": "audio_only"})
            return "", ""

        if not self.ocr_enabled and not self.vision_service.enabled:
            if self._ocr_unavailable_reason:
                logger.info("Slide OCR skipped: media=%s reason=paddle_unavailable", media_path)
            self._report(progress_cb, "ocr_sampling", 1.0, {"skipped": True})
            self._report(progress_cb, "ocr_inference", 1.0, {"skipped": True})
            self._report(progress_cb, "vision_inference", 1.0, {"skipped": True})
            return "", ""

        frames_dir = work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        self._sample_frames(media_path, frames_dir)
        total_frames = len(list(frames_dir.glob("*.jpg")))
        self._report(progress_cb, "ocr_sampling", 1.0, {"sampled_frames": total_frames})
        logger.info("Slide OCR frame sampling finished: media=%s frames=%d", media_path, total_frames)

        unique_frames = self._deduplicate_frames(sorted(frames_dir.glob("*.jpg")))
        logger.info(
            "Slide OCR frame dedup finished: media=%s unique_frames=%d threshold=%d",
            media_path,
            len(unique_frames),
            self.slide_change_threshold,
        )
        if not unique_frames:
            self._report(progress_cb, "ocr_inference", 1.0, {"unique_frames": 0})
            self._report(progress_cb, "vision_inference", 1.0, {"unique_frames": 0})
            return "", ""

        texts: list[str] = []
        ocr_engine = self._build_ocr_engine()
        if ocr_engine is None:
            self._report(progress_cb, "ocr_inference", 1.0, {"skipped": True})
            ocr_text = ""
        else:
            logger.info(
                "Slide OCR inference started: media=%s unique_frames=%d interval_seconds=30",
                media_path,
                len(unique_frames),
            )
            processed_frames = 0
            last_beat = time.monotonic()
            for frame_path in unique_frames:
                try:
                    result = ocr_engine.ocr(str(frame_path), cls=True)
                    processed_frames += 1
                    frame_progress = processed_frames / max(1, len(unique_frames))
                    if not result:
                        now = time.monotonic()
                        if now - last_beat >= 30:
                            logger.info(
                                "Slide OCR heartbeat: media=%s processed_frames=%d extracted_entries=%d stage_progress=%.1f%%",
                                media_path,
                                processed_frames,
                                len(texts),
                                frame_progress * 100.0,
                            )
                            self._report(
                                progress_cb,
                                "ocr_inference",
                                frame_progress,
                                {"processed_frames": processed_frames, "unique_frames": len(unique_frames)},
                            )
                            last_beat = now
                        continue
                    lines: list[str] = []
                    for item in result[0] or []:
                        txt = item[1][0] if item and len(item) > 1 else ""
                        if txt:
                            lines.append(txt)
                    if lines:
                        texts.append(" ".join(lines))
                    now = time.monotonic()
                    if now - last_beat >= 30:
                        logger.info(
                            "Slide OCR heartbeat: media=%s processed_frames=%d extracted_entries=%d stage_progress=%.1f%%",
                            media_path,
                            processed_frames,
                            len(texts),
                            frame_progress * 100.0,
                        )
                        self._report(
                            progress_cb,
                            "ocr_inference",
                            frame_progress,
                            {"processed_frames": processed_frames, "unique_frames": len(unique_frames)},
                        )
                        last_beat = now
                except Exception as exc:  # noqa: BLE001
                    logger.warning("OCR failed for %s: %s", frame_path, exc)

            self._report(progress_cb, "ocr_inference", 1.0, {"processed_frames": processed_frames})
            ocr_text = "\n".join(texts)

        self._report(progress_cb, "vision_inference", 0.0, {"unique_frames": len(unique_frames)})
        vision_descriptions = self.vision_service.describe_frames(unique_frames)
        self._report(
            progress_cb,
            "vision_inference",
            1.0,
            {"described_frames": min(len(unique_frames), self.vision_service.max_frames)},
        )
        vision_text = "\n".join(vision_descriptions)
        logger.info("Slide vision reasoning finished: media=%s descriptions=%d", media_path, len(vision_descriptions))
        return ocr_text, vision_text

    def _sample_frames(self, media_path: Path, frames_dir: Path) -> None:
        logger.info(
            "Slide OCR frame sampling started: media=%s fps=%.3f output_dir=%s",
            media_path,
            self.ocr_fps,
            frames_dir,
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(media_path),
            "-vf",
            f"fps={self.ocr_fps}",
            str(frames_dir / "frame_%06d.jpg"),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def _report(
        progress_cb: Callable[[str, float, dict[str, Any] | None], None] | None,
        stage: str,
        progress: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        if progress_cb:
            progress_cb(stage, progress, details)

    def _deduplicate_frames(self, frame_paths: list[Path]) -> list[Path]:
        unique: list[Path] = []
        prev_hash: str | None = None
        for frame_path in frame_paths:
            h = _dhash(frame_path)
            if prev_hash is None:
                unique.append(frame_path)
                prev_hash = h
                continue
            dist = _hamming(prev_hash, h)
            if dist >= self.slide_change_threshold:
                unique.append(frame_path)
                prev_hash = h
        return unique

    def _build_ocr_engine(self):
        try:
            from paddleocr import PaddleOCR

            # 'latin' covers English + Polish characters better than 'en'.
            lang = "latin" if any(l in {"pl", "en"} for l in self.ocr_languages) else "en"
            return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PaddleOCR unavailable, skipping OCR: %s", exc)
            return None

    @staticmethod
    def _segment_to_line(seg: dict) -> str:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()
        speaker = seg.get("speaker")
        prefix = f"[{start:.2f}-{end:.2f}]"
        if speaker:
            return f"{prefix} {speaker}: {text}"
        return f"{prefix} {text}"


def _dhash(path: Path, hash_size: int = 8) -> str:
    image = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    diff = []
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            diff.append(left > right)
    return "".join("1" if d else "0" for d in diff)


def _hamming(h1: str, h2: str) -> int:
    max_len = max(len(h1), len(h2))
    h1 = h1.ljust(max_len, "0")
    h2 = h2.ljust(max_len, "0")
    return sum(c1 != c2 for c1, c2 in zip(h1, h2, strict=True))


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _probe_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(proc.stdout.strip())
    except Exception:  # noqa: BLE001
        return 0.0
