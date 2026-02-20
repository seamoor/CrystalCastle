# Local Privacy-First Knowledge Indexer (Docker)

Local, offline-first knowledge indexing for audio/video/PDF/PPTX with semantic retrieval, summaries, and chat UX.

## Stack

- `qdrant`: vector database
- `processor`: Python ingestion + indexing + query API
- `ollama`: local LLM + vision model runtime
- `open-webui`: chat frontend connected to `processor` (`local-rag` model)

No paid/cloud API is required.

## What It Does

- Watches `/watch` and auto-processes supported files.
- Supports: `.mp4`, `.mkv`, `.mov`, `.mp3`, `.wav`, `.m4a`, `.pdf`, `.pptx`.
- Media pipeline:
  - audio extraction (`ffmpeg`)
  - speech-to-text (`faster-whisper`, timestamps)
  - optional diarization (`pyannote.audio`)
  - slide text OCR (`PaddleOCR`)
  - slide visual reasoning (`Ollama` VLM, e.g. `llava:7b`)
- Document pipeline:
  - PDF text extraction (`pypdf`)
  - PPTX text extraction (`python-pptx`)
- Post-processing:
  - chunking + embeddings
  - local summary/tag generation
- Retrieval/query:
  - metadata filters (`filename`, `tags`, date range)
  - source citations with timestamps
  - OpenAI-compatible `/v1/chat/completions`

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

### 1. Start everything

```bash
docker compose up -d --build
```

Endpoints:
- Open WebUI: `http://localhost:3000`
- Processor API: `http://localhost:8080`
- Qdrant: `http://localhost:6333`

### 2. Pull local models (one-time)

```bash
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull llava:7b
```

### 3. Drop files into `./watch`

Files are auto-queued and processed.

### 4. Query

```bash
curl -X POST http://localhost:8080/query \
  -H 'content-type: application/json' \
  -d '{"query":"Streszczenie szkolenia", "top_k": 6}'
```

### 5. Force reindex a file

```bash
curl -X POST http://localhost:8080/ingest \
  -H 'content-type: application/json' \
  -d '{"path":"/watch/SBC FIREWALL AND SECURITY.mp4","force":true}'
```

### 6. Run tests

```bash
cd processor
pytest
```

Or:

```bash
docker compose run --rm processor pytest
```

## Prompt Presets

Use these in Open WebUI.

Global search across all indexed data:

```text
Answer using all indexed sources. If unsure, say what is missing and cite sources.
Question: <your question>
```

Only one file:

```text
Search ONLY in indexed <FILE_NAME>. If no relevant indexed context exists, return NO_INDEXED_CONTEXT.
Question: <your question>
```

Evidence-first / low-risk response:

```text
Answer from indexed context only. Include key points with source filenames and timestamps.
If evidence is missing, return NO_INDEXED_CONTEXT.
Question: <your question>
```

## Configuration

Main file: `config/config.yml`.

Important keys:
- `whisper.model_size`
- `diarization.enabled`
- `ocr.enabled`, `ocr.fps`, `ocr.slide_change_threshold`
- `vision.enabled`, `vision.model`, `vision.max_frames`
- `embedding.model_name`
- `llm.enabled`, `llm.model`, `llm.timeout_seconds`
- `query.strict_grounding` (currently default `false`)

## Current Runtime Behavior

- On startup, watcher scans existing files in `/watch`.
- Files already marked `indexed` in state DB are skipped.
- If you used `force=true`, old entries are deleted and file is indexed again.

## Progress and Debug Endpoints

- `GET /debug/queue` -> queue size
- `GET /debug/worker` -> worker alive + queue
- `GET /debug/jobs` -> live job progress (stage + overall progress)
- `GET /debug/jobs/by-filename?filename=...`
- `GET /debug/filenames` -> indexed filenames

Progress includes stages like:
- `transcription`
- `ocr_sampling`
- `ocr_inference`
- `vision_inference`
- `summary_tagging`

Heartbeats are logged for long stages (STT/OCR/vision/summary-tagging).

## API Reference

- `GET /health`
- `POST /ingest` body: `{"path":"/watch/file.pdf"}`
- `POST /ingest` force: `{"path":"/watch/file.pdf", "force": true}`
- `POST /query` body: `{"query":"...", "top_k":8, "filters": {...}}`
- `GET /dashboard/stats`
- `GET /debug/*` endpoints above
- `GET /v1/models`
- `POST /v1/chat/completions`

## Open WebUI Notes

- Model exposed by processor is `local-rag`.
- Open WebUI is configured to use processor OpenAI-compatible API, not direct Ollama.
- If WebUI state gets inconsistent, reset volume:

```bash
docker compose down
docker volume rm crystalcastle_open_webui_data
docker compose up -d --build
```

## Metadata Stored in Qdrant

Per chunk:
- `doc_id`, `chunk_id`, `chunk_index`
- `filename`, `path`
- `text`, `summary`
- `date_indexed`, `language`, `tags`, `speakers`
- `duration_seconds`, `timestamp_start`, `timestamp_end`
- `page_start`, `page_end` (PDF)
- `slide_start`, `slide_end` (PPTX)

Note: slide/page source references appear for newly indexed documents. Reindex existing files to backfill these fields.

## Platform Notes

Windows + NVIDIA:

```bash
docker compose --profile nvidia up -d --build qdrant processor-nvidia open-webui ollama
```

macOS Apple Silicon:
- CPU mode in Docker works.
- Heavy steps (STT/vision) can be slow; tune model sizes and `top_k`.

## Troubleshooting

No OCR:
- check logs for `Paddle runtime detected. Slide OCR is enabled.`
- rebuild `processor` after dependency changes.

`Local LLM unavailable`:
- verify Ollama is up and model exists (`ollama list`)
- restart `processor` after fixes

Open WebUI request “hangs” but result appears after refresh:
- backend may have completed but UI stream render lagged
- reduce prompt size / `top_k`

Long response times:
- expected on CPU for large context and vision models
- use smaller models and tighter prompts when needed
