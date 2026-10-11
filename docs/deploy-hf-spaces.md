# Deploying FusionTrace to Hugging Face Spaces

FusionTrace runs on a Docker Space on the free **CPU basic** hardware (2 vCPU,
16 GB RAM). The Space builds the repository's `Dockerfile`; model weights are
downloaded from the Hugging Face Hub when the container starts.

This guide prepares and documents the deployment. Nothing in the repository
deploys automatically.

## What runs where

| Piece | Where it comes from |
|---|---|
| Code, `Dockerfile`, `requirements.txt` | Staged from a commit by `deploy/hf-spaces/prepare_space.sh` |
| Space card (`README.md`, `sdk: docker`, `app_port: 7860`) | `deploy/hf-spaces/README.md` |
| Model weights | Hub model repositories, fetched on start by `backend/scripts/download_models.py` |
| Secrets (`HF_TOKEN`, `GEMINI_API_KEY`, ...) | Space settings, injected as environment variables at runtime |

Weights are not baked into the image. That keeps `HF_TOKEN` out of image layers
and lets you switch detectors with a variable instead of a rebuild. The trade-off
is that every cold start downloads the weights again (see
[Storage is ephemeral](#storage-is-ephemeral)).

## 1. Prerequisites

- A Hugging Face account and the `hf` CLI (`pip install huggingface_hub`, then
  `hf auth login` with a token that has **write** access to the Space).
- The weights of every detector you will enable published on the Hub. Each
  detector declares its repositories in `weights()`; list them with
  `python scripts/download_models.py --dry-run` (from `backend/`). For a private
  weights repository, create a fine-grained **read** token scoped to it; it
  becomes the Space's `HF_TOKEN` secret.

## 2. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Pick a name, **Docker** as the SDK and the **Blank** template.
3. Hardware: **CPU basic** (free).
4. Visibility: start **Private**. A public Space lets anyone upload media to it;
   make it public only after reading [Privacy](#privacy).

## 3. Variables and secrets

Space → **Settings** → **Variables and secrets**. Secrets are hidden and not
copied when someone duplicates the Space; variables are visible.

| Name | Kind | Value | Needed when |
|---|---|---|---|
| `HF_TOKEN` | secret | read token for the weights repositories | any weights repository is private or gated |
| `HF_REPO_ID` | secret or variable | e.g. `your-user/FusionTrace-models` | a detector's `WeightsSpec` sets `repo_id_env="HF_REPO_ID"` (a private mirror); secret if the repo name is private |
| `GEMINI_API_KEY` | secret | Google AI Studio key | you want the optional AI summary; it sends detector scores, not media, to Google |
| `PORT` | variable | `7860` | optional: the staged `Dockerfile` already sets it; must match `app_port` |
| `FUSIONTRACE_VIDEO_ENABLED` | variable | `false` | memory is tight (see [Memory](#memory)) |
| `FUSIONTRACE_RETENTION_HOURS` | variable | e.g. `1` | recommended for a shared demo: how long uploads and artifacts are kept |
| `FUSIONTRACE_<KIND>_MODEL` | variable | a registry key | you want a non-default detector for `image`, `audio` or `video` |

API keys, rate limits and other application settings are listed in
`backend/.env.example`; set them here the same way (secrets for anything
sensitive).

## 4. Stage and upload

From the repository root, on the commit you want to deploy:

```bash
deploy/hf-spaces/prepare_space.sh            # writes .space-build/ (gitignored)
```

The script uses `git archive`, so only committed files are staged: no local
`backend/data`, weights or `.env`. It drops notebooks, tests and the sample media
under `backend/test/`, copies the Space card to `README.md`, and appends
`ENV PORT=7860` and `FORWARDED_ALLOW_IPS=*` (trust the Spaces proxy's client
address) to the staged `Dockerfile`.

Review `.space-build/`, then upload it:

```bash
hf upload your-user/your-space .space-build . --repo-type space
```

Alternatively clone the Space repository (`git clone https://huggingface.co/spaces/your-user/your-space`),
copy the staged files in, commit and push. Uploads add or replace files but do
not delete files you removed locally; delete those in the Space's **Files** tab.
For the next release, delete `.space-build/` and stage again.

## 5. First start

1. The **Build** logs show the image build (system packages, then the pinned,
   CPU-only Python dependencies).
2. The **Container** logs show `scripts/download_models.py` fetching each
   enabled detector's weights, then uvicorn listening on port 7860. The Space
   card sets `startup_duration_timeout: 1h` so a slow first download is not
   killed by the default startup timeout.
3. Open `https://your-user-your-space.hf.space/health`. Every enabled detector
   should report `"available": true`. A detector whose weights failed to download
   reports `false`; the entrypoint logs the reason and the app still starts.

## Memory

All enabled detectors are loaded into the container's RAM at startup (warm-up),
and the free tier has 16 GB. If the Space is restarted for exceeding its memory
limit:

1. Set `FUSIONTRACE_VIDEO_ENABLED=false` first (the temporal video detector is
   the newest and least essential signal; frame-level image analysis and the
   soundtrack's audio analysis still run on videos).
2. If that is not enough, pick smaller models through
   `FUSIONTRACE_<KIND>_MODEL`, or disable a kind entirely.
3. `FUSIONTRACE_WARMUP=0` lowers memory at boot by loading detectors on first
   use instead, at the cost of a slow first scan.

## Storage is ephemeral

Free Spaces have no persistent disk. On every restart, rebuild, or wake-up after
the Space has gone to sleep for inactivity:

- `/app/data` (scan history in SQLite, uploads, heatmaps) is empty again, so
  links to old reports stop working;
- `/app/models` is empty again, so weights are downloaded again and the first
  request after a cold start waits for the download and model load.

Hugging Face sells persistent storage as a paid upgrade (mounted at `/data`);
with it, set `FUSIONTRACE_DATA_DIR=/data/app-data` and
`FUSIONTRACE_MODEL_DIR=/data/models`. The free setup does not need it.

## Sample media

The Space repository is public whenever the Space is. Never commit real user
media (or any media you are not licensed to redistribute) to it.
`prepare_space.sh` deliberately excludes `backend/test/`. If you add example
files for visitors, use public, openly licensed media and credit the source.

## Privacy

- Uploaded files are processed and stored on Hugging Face's infrastructure
  for up to `FUSIONTRACE_RETENTION_HOURS` (and until the next restart). Keep the
  Space private, or keep retention short and enable the app's API key and rate
  limit settings, before sharing the link.
- With `GEMINI_API_KEY` set, the AI summary sends detector scores (not the
  media) to Google. Leave it unset for a self-contained Space.
- `HF_HUB_DISABLE_TELEMETRY=1` is set in the image.

## Troubleshooting

| Symptom | Check |
|---|---|
| Space stuck in "Starting" | Container logs: weight download progress or an error; `HF_TOKEN` access to the weights repos |
| `/health` shows `"available": false` | Container logs from `download_models.py`; repo id, revision and token |
| 404 / connection refused through the Space URL | `PORT` and `app_port` must both be 7860 |
| Space restarts under load | [Memory](#memory) |
| Every client shares one rate-limit bucket | `FORWARDED_ALLOW_IPS=*` missing from the staged `Dockerfile` |
