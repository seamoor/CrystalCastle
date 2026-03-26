from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class WhisperConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "int8"


class DiarizationConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    enabled: bool = False
    model_path: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class OCRConfig(BaseModel):
    enabled: bool = True
    fps: float = 0.2
    slide_change_threshold: int = 10
    languages: list[str] = Field(default_factory=lambda: ["en", "pl"])


class ChunkingConfig(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 160


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    normalize_embeddings: bool = True


class QueryConfig(BaseModel):
    strict_grounding: bool = False
    extractive_max_snippets: int = 8
    min_score: float = 0.25
    max_context_chars: int = 12000
    rerank_enabled: bool = True
    rerank_top_n: int = 12
    max_chunks_per_doc: int = 4


class VisionConfig(BaseModel):
    enabled: bool = True
    model: str = "llava:7b"
    base_url: str = "http://ollama:11434"
    max_frames: int = 12
    timeout_seconds: int = 180


class LLMConfig(BaseModel):
    enabled: bool = True
    model: str = "llama3.1:8b"
    base_url: str = "http://ollama:11434"
    timeout_seconds: int = 180


class QdrantConfig(BaseModel):
    host: str = "qdrant"
    port: int = 6333
    collection_name: str = "knowledge_chunks"


class ProcessorConfig(BaseModel):
    watch_dir: str = "/watch"
    data_dir: str = "/data"
    models_dir: str = "/models"
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mkv", ".mov", ".mp3", ".wav", ".m4a", ".pdf", ".pptx"]
    )
    gpu_enabled: bool = False
    poll_interval_seconds: float = 2.0


class AppConfig(BaseModel):
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: {path}")
    return data


def load_config() -> AppConfig:
    config_path = Path(os.getenv("CONFIG_PATH", "/app/config/config.yml"))
    env_overrides = {
        "processor": {
            "watch_dir": os.getenv("WATCH_DIR", "/watch"),
            "data_dir": os.getenv("DATA_DIR", "/data"),
            "models_dir": os.getenv("MODELS_DIR", "/models"),
            "gpu_enabled": os.getenv("GPU_ENABLED", "false").lower() == "true",
        },
        "qdrant": {
            "host": os.getenv("QDRANT_HOST", "qdrant"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
        },
        "llm": {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        },
        "vision": {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        },
    }

    file_cfg = _read_yaml(config_path)
    merged = file_cfg | env_overrides
    for section, values in env_overrides.items():
        if section in file_cfg:
            merged[section] = file_cfg[section] | values

    cfg = AppConfig.model_validate(merged)
    Path(cfg.processor.watch_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.processor.data_dir).mkdir(parents=True, exist_ok=True)
    return cfg
