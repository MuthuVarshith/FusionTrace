# FusionTrace — Technical Assessment

**Repository:** `MuthuVarshith/FusionTrace` · branch `main` @ `57803c1`
**Reviewed:** 2026-10-11
**Scope:** full read of all source, config, deployment and test files.

---

## 1. Executive summary

FusionTrace is a FastAPI monolith that scores uploaded images, audio and video for
synthetic-media risk using two local PyTorch models, fuses the signals into a weighted
risk score, and renders an evidence report with a printable A4 PDF.

**Size:** ~2,300 lines of hand-written application code (plus ~1,900 lines of training
notebooks), SQLite persistence, no frontend framework, 814 MB of model weights committed
via Git LFS.

**Overall read:** the *thinking* is better than the *engineering*. The evidence-fusion
framing, the explainability work and the consistent refusal to over-claim are exactly what
a forensics tool needs, and they are genuinely well executed. But the project is a
single-user demo wearing production clothing: the README promises a Docker Compose setup
that does not start, an AI summary that cannot fire, and privacy on an API with no
authentication. None of this is unfixable — the defects are concentrated in wiring,
concurrency and packaging, not in the core design.

---

## 2. Architecture as implemented

```
POST /api/scans  →  stream to disk (1 MB chunks, size-capped)
                 →  content validation (PIL verify / soundfile / ffmpeg decode)
                 →  SQLite row, status=queued, return 202
                 →  FastAPI BackgroundTask: process_scan
                         ├ image: sha256 + EXIF + EfficientNetV2-L
                         ├ audio: sha256 + Wav2Vec2 (first 4 s only)
                         └ video: ffmpeg → ≤12 frames @ 1/5 s → per-frame image model
                                  + ffmpeg audio strip → Wav2Vec2
                 →  fuse_evidence() → status=completed          (Phase 1)
                 →  _enrich_scan(): GradCAM heatmap + Gemini summary  (Phase 2)
GET  /api/scans/{id}  ← browser polls every 2 s, up to 120 attempts
```

| Layer | Implementation |
|---|---|
| API | FastAPI, 9 routes, `BackgroundTasks` for async work |
| Image model | `tf_efficientnetv2_l` (timm), custom head, sigmoid ≥ 0.5 = Fake |
| Audio model | `Wav2Vec2ForSequenceClassification`, 16 kHz mono, 4 s window |
| Video | FFmpeg frame sampling + per-frame image model + extracted-audio model |
| Explainability | GradCAM from raw PyTorch hooks, NumPy jet colormap, PIL blend |
| Fusion | Weighted mean of per-signal risk → `low` / `medium` / `high` |
| Persistence | SQLite, JSON report column, self-migrating schema |
| Frontend | Vanilla JS (622 lines), polls for status, client-side PDF via print iframe |
| LLM | Google Gemini, optional, summarises detector signals only |

The two-phase design — commit the verdict first, run enrichments after — is genuinely
good: the UI is never blocked on GradCAM or a network LLM call.

---

## 3. What is well done

- **The evidence model is honest.** `fuse_evidence` (`backend/app/service.py:55`) emits
  per-signal risk, weight and a human-readable basis string, and every surface repeats
  that the result is not proof of authenticity. That framing is the project's real
  differentiator and it is applied consistently across API, UI and PDF.
- **Content validation over extension trust** (`backend/app/service.py:132`) — PIL
  `verify()`, soundfile header probe, FFmpeg decode. Most projects of this kind trust the
  user-controlled filename.
- **GradCAM implemented from scratch** (`backend/app/image_detection.py:109`) with raw
  PyTorch hooks and a NumPy jet colormap — no OpenCV, no `pytorch-grad-cam`. The maths is
  correct and the hooks are removed in a `finally` block.
- **Lazy, pre-warmed model caches**, streaming upload with a hard size cap,
  retention-based cleanup of uploads and artifacts.
- **XSS-safe frontend.** `escapeHtml` is applied at every interpolation point in both the
  on-screen render and the generated PDF.
- **UUID-typed path parameters** close off path traversal on the `/heatmap` route.
- **Schema self-migration** (`PRAGMA table_info` + `ALTER TABLE`) and a `CHECK` constraint
  on feedback labels.
- **The AI summary prompt is well constrained** (`backend/app/ai_summary.py:22`): hedged
  language enforced, raw confidence numbers stripped from the payload, no action
  recommendations, graceful `None` on failure.

---

## 4. Critical defects

### 4.1 The AI forensic summary is dead code

`backend/app/main.py:52` declares `create_scan(background_tasks, file)` — it never reads a
`generate_ai_summary` form field, so `service.create_scan` always takes its `False`
default. The frontend does send the field
(`backend/templates/static/js/script.js:51`), but there is **no checkbox with id
`generate-ai-summary` anywhere in `index.html`**, so it always sends `'false'` regardless.

Consequence: `ai_summary_status` is permanently `"not_requested"`, the Gemini path never
executes, and the README feature table, the API reference, the PDF generator and the
`pollEnrichments` loop all describe a feature no user can trigger. Two independent breaks
in the same chain.

### 4.2 `docker compose up` produces an unreachable app

The Dockerfile was retargeted to HuggingFace Spaces — `CMD uvicorn ... --port 7860` — but
`docker-compose.yml` still maps `8000:8000`. The container listens on 7860 and nothing is
published there. README "Option B — Docker Compose (One Command)" does not work.

### 4.3 Model objects are shared mutable state across threads

`_get_image_model()` returns one cached module. GradCAM calls `model.zero_grad()`,
registers hooks and runs `backward()` on it (`backend/app/image_detection.py:134`) while
another background task may be inside `detect_image_deepfake` on the same object.
Concurrent scans can corrupt each other's gradients or capture the wrong activations.

Neither cache's lazy initialisation is lock-guarded, so the startup pre-warm thread and a
first inbound request can both load the model — a transient 2× memory spike for a
~500 MB EfficientNetV2-L.

### 4.4 Image inference failure degrades into a confusing scan failure

`detect_image_deepfake` catches every exception and returns
`{"prediction": "Error", "confidence": str(e)}` (`backend/app/image_detection.py:104`).
The caller immediately runs `float(result["confidence"].rstrip("%"))`
(`backend/app/service.py:211`), which raises `ValueError` on that message. The scan then
fails with a string-parse error that hides the actual cause.

### 4.5 Raw exception strings are persisted and served to clients

`backend/app/service.py:199` stores `str(exc)` in `scans.error`, returned verbatim by
`GET /api/scans/{id}`. Filesystem paths, model paths and FFmpeg stderr leak to any caller.

### 4.6 No authentication, and `GET /api/scans` is global

Any visitor receives every scan's filename, media type, risk rating and ID — and with an
ID, the full report including SHA-256, EXIF and the GradCAM heatmap. For a tool whose
headline claim is privacy, any shared deployment is a cross-tenant leak. Feedback
submission is equally unauthenticated and can be spammed.

---

## 5. Significant weaknesses

| Area | Issue |
|---|---|
| **Concurrency** | `BackgroundTasks` runs CPU-bound inference in Starlette's threadpool with no queue, no concurrency cap and no backpressure. A handful of 200 MB videos saturates the host. SQLite is opened without WAL or a raised `timeout`, so `database is locked` is reachable under load. |
| **Validation DoS** | Video and MP3 validation full-decodes untrusted media (`ffmpeg -map 0 -f null -`) with a 30 s timeout. A legitimate large video will time out and be rejected; a crafted one burns CPU. `ffprobe` metadata inspection would be the right tool. |
| **Audio ceiling** | Every clip is hard-trimmed to **4 seconds** (`backend/app/audio_detection.py:55`) because that is the training window. A 10-minute call is judged on its first 4 seconds. Disclosed in the README, but it is the single biggest limit on real-world validity. |
| **Fusion calibration** | Weights (0.1 / 0.8 / 0.9) are hardcoded and uncalibrated. `file-inspection` always contributes `risk 0.0, weight 0.1`, which structurally caps the maximum achievable score at 90% and pulls every score down by roughly 10%. No reliability diagrams, no threshold tuning. |
| **Disk growth** | `preprocess_audio` writes a `converted_*.wav` into `ARTIFACT_DIR` for every audio scan and never removes it; video frame directories likewise persist. Only the 24-hour retention sweep reclaims them, and that sweep is mtime-based, so a long-running scan's inputs can be deleted mid-flight. |
| **Tests** | 98 lines covering `fuse_evidence`, `media_type` and the feedback/history DB layer. Zero coverage of any HTTP endpoint, the video pipeline, GradCAM, upload validation or the enrichment phase. No `.github/workflows` — no CI at all. |
| **Dependency hygiene** | Four overlapping manifests: root `requirements.txt`, `backend/requirements.txt`, root `pyproject.toml` (`name = "fusiontrace"`), `backend/pyproject.toml` (`name = "mdfd"`, different version floors), plus a `uv.lock` nothing installs from. All constraints are `>=`, so builds are not reproducible — a `torch` or `transformers` minor release can break the image silently. |
| **Repo weight** | 814 MB of weights in LFS makes `git clone` an 813 MB download. `scripts/download_models.py` already exists to fetch from HF Hub; the weights should not also live in history. |
| **Dead code** | `backend/templates/upload.html` is an 83-line fully commented-out file. `backend/requirements_working.txt` is a stale artifact. |
| **Ops** | No structured logging, no request IDs, no metrics, no rate limiting. `logging.basicConfig` is called twice (`config.py` and `image_detection.py`). `__import__("os")` inline in the CORS middleware (`backend/app/main.py:31`) is a smell in otherwise clean code. |
| **Portability** | `config.py` copies the imageio FFmpeg binary to `ffmpeg.exe` — Windows-specific logic that runs unconditionally, including inside Linux containers. |
| **UI gap** | The GradCAM heatmap is generated and served, and appears in the exported PDF, but is never displayed in the web UI. |

---

## 6. Upgrade path

### Tier 1 — make the advertised product true
Small, high-value, one branch. This is the gap between what the README claims and what runs.

1. Wire `generate_ai_summary` through the endpoint **and** add the checkbox to `index.html`.
2. Fix the compose port; align Dockerfile and compose on one port with an env override.
3. Lock-guard both model caches; give GradCAM its own model instance or a mutex.
4. Fix the `"Error"` / `confidence` crash path; stop returning raw exceptions to clients.
5. Display the GradCAM heatmap in the web UI.

### Tier 2 — production hardening

6. API-key authentication; scope `GET /api/scans` to the caller; add rate limiting.
7. SQLite WAL + busy timeout, or move to Postgres; add a real job queue with a concurrency cap.
8. Replace full-decode validation with `ffprobe` metadata inspection.
9. Collapse to one dependency manifest with pinned versions; drop weights from LFS.
10. CI: lint, typecheck and tests on push.

### Tier 3 — substance

11. Chunked audio analysis (sliding 4 s windows, aggregated) to lift the 4-second ceiling.
12. Calibrate fusion weights against a held-out set; publish reliability curves.
13. Endpoint and pipeline tests, including a synthetic video fixture.
14. Temporal / lip-sync signal for video — the roadmap item that would most change the product.

---

## 7. File inventory

| Path | Lines | Role |
|---|---|---|
| `backend/app/main.py` | 114 | FastAPI app, 9 routes, lifespan |
| `backend/app/service.py` | 446 | Scan lifecycle, evidence fusion, persistence |
| `backend/app/image_detection.py` | 190 | EfficientNetV2 inference + GradCAM |
| `backend/app/audio_detection.py` | 183 | Wav2Vec2 inference + audio preprocessing |
| `backend/app/ai_summary.py` | 96 | Gemini forensic summary |
| `backend/app/config.py` | 59 | Settings, paths, env loading |
| `backend/templates/index.html` | 324 | Single-page UI |
| `backend/templates/static/js/script.js` | 622 | Upload, polling, render, PDF export |
| `backend/templates/static/css/styles.css` | 1,040 | Styling |
| `backend/tests/test_service.py` | 98 | Unit tests (fusion, feedback, history) |
| `backend/scripts/evaluate_models.py` | 87 | Labelled-folder benchmark |
| `backend/scripts/download_models.py` | 55 | HF Hub weight download |
| `backend/notebook/**` | ~1,900 | Training notebooks and dataset code |
