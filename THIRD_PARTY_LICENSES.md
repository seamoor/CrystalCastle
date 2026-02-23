# Third-Party Licenses (CrystalCastle)

Last updated: 2026-02-23

This document summarizes licenses for the software stack used in this repository.
It is for internal review and is not legal advice.

## 1) Runtime Products (Docker Compose)

| Component | Version | License | Notes |
|---|---|---|---|
| Qdrant | 1.13.4 | Apache-2.0 | Vector database |
| Ollama (server/image) | 0.5.7 | MIT | Model runtime; model licenses are separate |
| Open WebUI | main | Open WebUI License | Image label reports `NOASSERTION`; verify repo license terms before production use |
| Docker Desktop (host runtime) | n/a | Docker Subscription Service Agreement | Commercial usage depends on org size/revenue conditions |

## 2) Model Licenses (Critical)

| Model | License | Notes |
|---|---|---|
| llama3.1:8b | LLAMA 3.1 COMMUNITY LICENSE AGREEMENT | Review allowed use/distribution clauses |
| llava:7b | Apache-2.0 | Vision model used by visual reasoning pipeline |

## 3) Python Dependencies (Top-Level)

From `processor/requirements.txt`:

| Package | License |
|---|---|
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| pydantic | MIT |
| PyYAML | MIT |
| qdrant-client | Apache-2.0 |
| sentence-transformers | Apache-2.0 |
| numpy | BSD-style (NumPy license) |
| pypdf | BSD |
| python-pptx | MIT |
| watchdog | Apache-2.0 |
| requests | Apache-2.0 |
| python-multipart | Apache-2.0 |
| faster-whisper | MIT |
| Pillow | HPND |
| opencv-python-headless | Apache-2.0 |
| paddlepaddle | Apache-2.0 |
| paddleocr | Apache-2.0 |
| scikit-image | BSD (primarily BSD-3-Clause) |
| python-dotenv | BSD-3-Clause |
| pyannote.audio | MIT |

## 4) System Binaries / Native Tooling

| Component | License Status | Notes |
|---|---|---|
| ffmpeg (inside processor image) | GPL-enabled build | Detected `--enable-gpl` in build config; treat as GPL build for compliance review |

## 5) Compliance Checklist (for Legal / Security Review)

1. Confirm organization is eligible for Docker Desktop free/commercial terms, or purchase required subscription.
2. Approve Open WebUI license terms (branding/redistribution clauses may apply).
3. Approve LLAMA 3.1 model license for intended internal/commercial use.
4. Confirm policy for GPL-enabled ffmpeg build in your distribution/deployment model.
5. Generate and archive SBOM for each release (`processor`, `open-webui`, `ollama` images).
6. Keep a copy of all third-party license texts with release artifacts.
7. Re-run license audit whenever dependency or model versions change.

## 6) How This Was Collected

- Docker image metadata and in-container inspection.
- Local model metadata via `ollama show`.
- Installed package metadata from `importlib.metadata` inside `processor` container.

## 7) Verification Commands

```bash
# Compose components
docker image inspect qdrant/qdrant:v1.13.4 --format '{{json .Config.Labels}}'
docker image inspect ollama/ollama:0.5.7 --format '{{json .Config.Labels}}'
docker image inspect ghcr.io/open-webui/open-webui:main --format '{{json .Config.Labels}}'

# Model licenses
docker exec ollama ollama show llama3.1:8b
docker exec ollama ollama show llava:7b

# ffmpeg licensing hints
docker compose run --rm processor ffmpeg -version | head -n 3

# Python dependency license metadata
docker compose run --rm processor python -c "from importlib.metadata import metadata; print(metadata('fastapi').get('License'))"
```
