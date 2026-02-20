from pathlib import Path

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".mp3", ".wav", ".m4a"}
PDF_EXTENSIONS = {".pdf"}
PPTX_EXTENSIONS = {".pptx"}


def classify_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in MEDIA_EXTENSIONS:
        return "media"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in PPTX_EXTENSIONS:
        return "pptx"
    return "unsupported"
