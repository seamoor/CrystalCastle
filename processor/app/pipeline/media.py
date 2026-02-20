from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from app.pipeline.diarization import DiarizationService, align_speakers

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

    def process(self, path: Path) -> dict:
        work_dir = self.data_dir / "jobs" / path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Media processing started: path=%s work_dir=%s", path, work_dir)

        audio_path = work_dir / "audio.wav"
        logger.info("Extracting audio track: input=%s output=%s", path, audio_path)
        self._extract_audio(path, audio_path)

        logger.info("Transcription started: audio=%s", audio_path)
        transcript = self._transcribe(audio_path)
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
        slide_text = self._extract_slides_text(path, work_dir)
        logger.info("Slide OCR finished: extracted_chars=%d", len(slide_text))
        transcript_text = "\n".join(
            self._segment_to_line(seg) for seg in segments
        )

        merged_text = transcript_text
        if slide_text:
            merged_text += "\n\n[SLIDES OCR]\n" + slide_text

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

    def _transcribe(self, audio_path: Path) -> dict:
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
        segments, info = model.transcribe(str(audio_path), vad_filter=True, word_timestamps=False)

        out_segments: list[dict] = []
        for seg in segments:
            out_segments.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                }
            )

        return {
            "language": getattr(info, "language", None),
            "segments": out_segments,
        }

    def _extract_slides_text(self, media_path: Path, work_dir: Path) -> str:
        if not self.ocr_enabled or media_path.suffix.lower() in {".mp3", ".wav", ".m4a"}:
            if self._ocr_unavailable_reason:
                logger.info("Slide OCR skipped: media=%s reason=paddle_unavailable", media_path)
            return ""

        frames_dir = work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        self._sample_frames(media_path, frames_dir)
        total_frames = len(list(frames_dir.glob("*.jpg")))
        logger.info("Slide OCR frame sampling finished: media=%s frames=%d", media_path, total_frames)

        unique_frames = self._deduplicate_frames(sorted(frames_dir.glob("*.jpg")))
        logger.info(
            "Slide OCR frame dedup finished: media=%s unique_frames=%d threshold=%d",
            media_path,
            len(unique_frames),
            self.slide_change_threshold,
        )
        if not unique_frames:
            return ""

        texts: list[str] = []
        ocr_engine = self._build_ocr_engine()
        if ocr_engine is None:
            return ""

        for frame_path in unique_frames:
            try:
                result = ocr_engine.ocr(str(frame_path), cls=True)
                if not result:
                    continue
                lines: list[str] = []
                for item in result[0] or []:
                    txt = item[1][0] if item and len(item) > 1 else ""
                    if txt:
                        lines.append(txt)
                if lines:
                    texts.append(" ".join(lines))
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR failed for %s: %s", frame_path, exc)

        return "\n".join(texts)

    def _sample_frames(self, media_path: Path, frames_dir: Path) -> None:
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
