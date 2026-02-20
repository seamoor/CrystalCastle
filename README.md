# Local Privacy-First Knowledge Indexer (Docker)

Fully local knowledge indexing stack for audio/video/PDF/PPTX with semantic search and chat UX.

## Stack

- `qdrant`: vector database
- `processor`: Python ingestion + NLP pipeline + API
- `open-webui`: chat-style frontend (OpenAI-compatible target set to `processor`)
- `ollama` (optional profile): local LLM for summaries/tags/answers

All services run locally in Docker. No paid/cloud API is required.

## Features

- Offline-first architecture
- Watch folder ingestion from `/watch`
- Supported types: `mp4`, `mkv`, `mov`, `mp3`, `wav`, `m4a`, `pdf`, `pptx`
- Media pipeline:
  - audio extraction via `ffmpeg`
  - transcription via `faster-whisper` (timestamps, EN/PL)
  - optional diarization via `pyannote.audio`
  - video slide OCR via frame sampling + dedup + PaddleOCR
- Document pipeline:
  - PDF text extraction (`pypdf`)
  - PPTX slide text extraction (`python-pptx`)
- Post-processing:
  - semantic chunking
  - multilingual embeddings (`sentence-transformers`)
  - local LLM summaries + tags via Ollama
- Vector store:
  - Qdrant collection auto-init
  - rich metadata (filename, indexed date, language, tags, speakers, timestamps)
- Query:
  - semantic retrieval with metadata filters
  - source citations with timestamps
  - OpenAI-compatible `/v1/chat/completions` for Open WebUI
- Dashboard API:
  - document count
  - total indexed duration
  - tag distribution
  - recent uploads

## Project Layout

```text
.
├── docker-compose.yml
├── config/
│   └── config.yml
├── processor/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── watcher.py
│       ├── config.py
│       ├── pipeline/
│       ├── services/
│       └── storage/
├── watch/
└── models/
```

## Quick Start

### 1. Start core services (CPU)

```bash
docker compose up -d --build
```

Open UI: `http://localhost:3000`

Processor API: `http://localhost:8080`

Qdrant: `http://localhost:6333`

If Open WebUI keeps routing prompts directly to Ollama instead of the local RAG API, reset its persisted state:

```bash
docker compose down
docker volume rm crystalcastle_open_webui_data
docker compose up -d --build
```

### 2. Load a local Ollama model (one-time per model)

```bash
docker exec -it ollama ollama pull llama3.1:8b
```

### 3. Drop files into watch folder

Put files into `./watch` on host. They appear as `/watch` in container and are auto-indexed.

### 4. Query

- Open WebUI chat points to `processor` OpenAI-compatible endpoint.
- Direct API query:

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query":"Streszczenie szkolenia", "top_k": 6, "filters": {"tags": ["szkolenie"]}}'
```

### 5. Dashboard API

```bash
curl http://localhost:8080/dashboard/stats
```

### 6. Run tests

```bash
cd processor
pytest
```

Or in Docker:

```bash
docker compose run --rm processor pytest
```

## Configuration

Main config: `config/config.yml`

Key settings:

- `diarization.enabled`: enable/disable speaker diarization
- `ocr.fps`: frame sampling rate for video OCR
- `whisper.model_size`: e.g. `tiny`, `base`, `small`, `medium`
- `chunking.chunk_size` / `chunking.chunk_overlap`
- `embedding.model_name`
- `processor.gpu_enabled`

Environment vars in compose can override host/paths.

## Windows + NVIDIA (RTX A1000 4GB)

Use GPU profile service:

```bash
docker compose --profile nvidia up -d --build qdrant processor-nvidia open-webui
```

Notes:

- Ensure Docker Desktop + NVIDIA Container Toolkit integration is enabled.
- For 4GB VRAM, use smaller whisper model (`base` or `small`) and keep batch sizes low.
- You can keep `compute_type: int8` for stability on constrained VRAM.

## macOS Apple Silicon

CPU mode is default and works on arm64 images.

```bash
docker compose up -d --build
```

Notes:

- `faster-whisper` runs in CPU mode unless GPU acceleration is available in your runtime.
- Keep `whisper.model_size` modest (`small`) for better throughput.

## Privacy + Offline Considerations

- No cloud APIs are called by default.
- Ollama runs locally; set `llm.enabled: false` if you want retrieval-only mode.
- For fully offline diarization, point `diarization.model_path` to local model files under `./models`.
- For fully offline model usage, pre-download required models into `./models` and avoid first-run pulls.

## API Reference

- `GET /health`
- `POST /ingest` body: `{"path":"/watch/file.pdf"}`
- `POST /ingest` force reprocess: `{"path":"/watch/file.pdf", "force": true}`
- `POST /query` body: `{"query":"...", "top_k":8, "filters": {"filename":"x", "tags":["a"], "date_from":"2026-01-01"}}`
- `GET /dashboard/stats`
- `GET /v1/models` (OpenAI-compatible)
- `POST /v1/chat/completions` (OpenAI-compatible for Open WebUI)

## Metadata Stored in Qdrant

Each chunk stores:

- `doc_id`, `chunk_id`, `chunk_index`
- `filename`, `path`
- `text`, `summary`
- `date_indexed`
- `language`
- `speakers`
- `tags`
- `duration_seconds`
- `timestamp_start`, `timestamp_end`

## Operational Notes

- Designed for batch ingestion with queue-based watcher worker.
- State tracking in SQLite at `/data/state.db`.
- Persistent volumes configured for Qdrant, processor data, Open WebUI, and Ollama.

## Troubleshooting

- If OCR or diarization dependencies fail, pipeline logs warning and continues where possible.
- If Ollama is not running, summaries/tags/answers may be empty unless `llm.enabled` is set to `false`.
- To enable slide OCR, rebuild `processor` after dependency changes:

```bash
docker compose up -d --build processor
docker compose logs -f processor
```

- OCR is active when logs include `Paddle runtime detected. Slide OCR is enabled.` and `Slide OCR finished: extracted_chars=...`.
- Check logs:

```bash
docker compose logs -f processor
```

## Next Hardening Steps (optional)

- Add retry/backoff queue persistence (Redis/RQ/Celery).
- Expand integration tests for end-to-end media + OCR pipelines.
- Add a dedicated metadata collection for document-level aggregates.
- Add auth/TLS for production LAN exposure.
