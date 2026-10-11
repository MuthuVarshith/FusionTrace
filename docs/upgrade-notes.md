# FusionTrace v2 — upgrade notes

Branch `upgrade/v2`, based on `main` @ `57803c1`. Built 2026-10-11 by six parallel
workstreams against one shared detector interface, then merged by the lead. Not merged into
`main`.

The defects this upgrade set out to fix are listed in
[assessment-2026-10-11.md](assessment-2026-10-11.md).

## Why

1. **The image model didn't generalize.** It scored over 95% on its own data but labelled
   images from other datasets as Real. The quick check reproduced this: fake recall 0.011 on
   90 public diffusion-generated faces.
2. **Video was frame-by-frame only.** Problems between frames, such as flicker, unnatural
   motion or lip-sync mismatch, were invisible.
3. **Audio scored only the first 4 seconds** of a clip.
4. **Advertised features didn't work,** and production basics were missing:
   - the AI summary could not be triggered
   - Docker Compose published the wrong port
   - there was no auth and every scan was visible to everyone
   - raw errors leaked to clients
   - scans ran in unbounded background tasks
   - concurrent heatmap generation shared model state
   - weights were stored in git LFS

## What changed

### Architecture: one detector contract

- `backend/app/detectors/base.py` defines `Detector` with:
  - lazy, lock-guarded `load()`
  - `analyze()` that is safe to call concurrently
  - optional `explain()` that keeps its own state per call
  - export hooks for ONNX
  - typed errors, each with a client-safe `public_message`
- `registry.py` selects a detector per media kind from `config.yaml`, with env overrides
  (`FUSIONTRACE_<KIND>_MODEL`, `_ENABLED`, `_WEIGHT`).
- `pipeline.py` routes each upload to its signals. Every evidence item now records the
  model that produced it and its status (`ok` / `disabled` / `error` / `skipped`).
- If no detector produces a usable signal, the scan **fails** instead of reporting a
  misleading "low".

### Models

| Signal | v1 | v2 default | Options |
|---|---|---|---|
| Image | EfficientNetV2-L (custom) | **Community Forensics ViT-S/16-384** (MIT, 87 MB) | GenD CLIP ViT-L/14, SigLIP deepfake v1, legacy EfficientNetV2 |
| Audio | Wav2Vec2-base, first 4 s | Same model, **sliding 4 s windows over the whole clip**, top-3-window mean | Two Hub Wav2Vec2 detectors (both worse) |
| Video | Frames + soundtrack | Frames + soundtrack + **temporal consistency heuristic** (weight 0.3) | — |

Fusion weights for image (0.9), audio (0.9), video frames (0.8) and soundtrack (0.9) are
unchanged. Quick-check numbers and every rejected candidate: [model-choice.md](model-choice.md).

### Product and API (workstream C)

- **AI summary:** an opt-in checkbox, off by default and disabled without `GEMINI_API_KEY`;
  the model id is set by `GEMINI_MODEL`. Only detector outputs are sent: never the media,
  its hash, or its EXIF.
- **Errors:** clients get generic messages; full errors go to the server log. Old raw error
  text in the database is scrubbed by a migration.
- **UI, report and PDF:**
  - each signal shows its model name, a heuristic badge where relevant, and its status
  - the image heatmap now appears on the page, not only in the PDF
- **Auth:** optional API keys (`FUSIONTRACE_API_KEYS`). Scans store a hash of the key that
  created them and are visible only to that key. Uploads and feedback are rate-limited (429).
- **Queue:** a bounded in-process scan queue with a worker cap (default 1). SQLite runs in
  WAL mode with a busy timeout. Scans interrupted by a restart are marked failed, not
  re-queued, so a file that crashes the server can't crash it again in a loop.
- **Validation:** audio and video are checked with `ffprobe` (stream types, duration limits)
  instead of a full decode. Images get PIL verify plus a pixel cap against decompression bombs.
- **Clean-up:**
  - per-scan work dirs are deleted when a scan finishes
  - the retention sweep skips scans that are still in progress
  - removed: the duplicate logging setup, the Windows ffmpeg binary copy, `upload.html`, the
    unused `/image/detect` and `/audio/detect` routes, and `requirements_working.txt`

### Packaging and deployment (workstream D)

- **Dependencies:**
  - one manifest (`pyproject.toml`) with exact pins and `dev`/`mlops` extras
  - `uv.lock`, plus a hash-pinned `requirements.txt` generated from it
  - CPU-only torch
  - pydub, audioop-lts and imageio-ffmpeg removed
- **Docker:** serves on `$PORT` (default 8000) as a non-root user, with a healthcheck. Weights
  download at container start into a volume, never at build time, so `HF_TOKEN` stays out of
  the image. Compose publishes on loopback.
- **Weights out of git:** `backend/models/` is untracked and gitignored, and
  `scripts/download_models.py` fetches each enabled detector's pinned weights from the Hub.
  The LFS history purge is written up but **not executed**: [lfs-cleanup.md](lfs-cleanup.md).
- **HF Spaces:** a deployment kit in `deploy/hf-spaces/`, with steps in
  [deploy-hf-spaces.md](deploy-hf-spaces.md).
- **CI:** GitHub Actions checks the lock file, runs ruff and pytest, and builds and
  smoke-tests the Docker image.

### MLOps (workstream E)

- ONNX export with a parity check, INT8 quantization (dynamic and static), and a benchmark
  comparing torch, ONNX FP32 and INT8 on latency, memory and size.
- `evaluate_models.py` now selects detectors through the registry and reports AUC, precision,
  recall, F1 and a confusion matrix.
- Optional MLflow tracking and a local model registry with candidate/production aliases.

See [mlops.md](mlops.md). These tools were only smoke-tested on a tiny test model; nothing
has been benchmarked on the real detectors yet.

## Decisions made without asking (and why)

- **Public samples were added to the quick checks.** No ~200-image labelled set exists
  locally, and the local test folder may overlap the v1 training data, so it cannot show a
  generalization failure. Public out-of-distribution samples are pinned by revision and
  reported separately from the local results.
- **Temporal weight is 0.3, not the planned 0.7.** The heuristic scored every quick-check clip
  at zero risk. At 0.7 it would pull every video score toward "low".
- **The audio default didn't change.** Both candidates failed the switching rule, which was
  set before the run.
- **The legacy weights stay private.** The default audio model only downloads when
  `HF_REPO_ID` points at your mirror. A fresh Docker or Spaces deploy without the mirror
  reports audio as unavailable on `/health`.
- **One app process.** The queue and rate limits are in memory.

## Verified during the upgrade

- 74 tests across the merged branch: 71 passed and 3 skipped before the image weights were
  downloaded; with the weights present, the image tests pass. `ruff check .` is clean.
- `uv lock --check` passes.
- End-to-end on a local server, with the default models loaded from `/health`:

| Upload | Rating (score) | Signals |
|---|---|---|
| StyleGAN fake image | HIGH (89.9) | image model 0.999; heatmap served |
| Fake WAV | HIGH (90.0) | audio model 1.0 |
| 10 s real-face video with a real voice | LOW (22.2) | frames 0.0, soundtrack 0.519, temporal 0.0 |

## Not done / open

- **Not run:** `docker build` (no Docker daemon was running) and the CI workflow (never
  pushed).
- **Temporal heuristic:** needs calibration, or a permissively licensed learned temporal or
  lip-sync model.
- **Image tests:** only diffusion faces; face swaps are untested. GenD would benefit from a
  face detector, which FusionTrace doesn't have.
- **Fusion:** the fixed 0-risk file-inspection signal still caps scores at about 90
  (unchanged from v1, as instructed).
- **Video evaluation:** `evaluate_models.py` scores the temporal detector without the
  pipeline's frame results.
- **MLOps tools:** exercised only on the tiny test model.
- **Ruff:** `pyproject.toml` has per-file ignores for some merged files; tighten them as those
  files are cleaned.

## Five interview questions you should be able to answer

1. **Your v1 image model scored over 95% but failed on other datasets. Why does that happen,
   and how did you show the new model is better rather than just different?**
   Shortcut learning on one dataset's artifacts, plus a normalization mismatch between
   training and inference. Evidence: an out-of-distribution sample kept separate from the
   local test set, pinned by revision, with AUC and both recalls reported. Also be ready to
   name the limit of that evidence: diffusion faces only.
2. **Why score audio as the mean of the top 3 windows instead of the mean or the max?**
   Mean dilutes a short spliced fake to nothing (0 of 10 spliced clips caught in the splice
   test). Max is triggered by one noisy window. Top-k caught the most splices (8 of 10) while
   still passing 9 of 10 real recordings (n = 20).
3. **How do you stop one model's mistake from deciding the result, and what does the risk
   score mean?**
   It's a weighted mean of independent signals, each reported with its model and status. A
   scan with no usable signal fails instead of reporting low. The score is a review-priority
   signal, not a probability that the media is fake.
4. **How does concurrency work here, and where would it break at scale?**
   - lock-guarded lazy model loading
   - concurrent-safe inference
   - explanations with per-call gradients and no shared hooks
   - a bounded queue with a worker cap, and SQLite in WAL mode

   It breaks across processes: the in-memory queue and rate limiter don't share state, and
   SQLite allows a single writer. Next steps would be Postgres plus a real queue (Redis or
   SQS) with model-serving workers.
5. **Why did you ship a heuristic for temporal video detection, and how would you make it
   trustworthy?**
   No candidate passed both the licence and safety bars. The heuristic is labelled as one,
   has a low weight, and failed its quick check openly (AUC 0.50). The path forward:
   calibrate its scaling on a labelled set with `evaluate_models.py`, add lip-sync checks,
   and replace it when a permissively licensed temporal model exists.
