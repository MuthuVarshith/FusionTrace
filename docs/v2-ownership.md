# FusionTrace v2 — workstreams and file ownership

Base commit: the commit that adds this file on `upgrade/v2`. Six workstreams run in
parallel, each on its own branch and worktree. **No two workstreams edit the same
file.** If you need a change in a file you don't own, describe it in your final
report; the lead applies it at merge time.

## Shared contract (lead-owned, frozen)

| File | Purpose |
|---|---|
| `backend/app/detectors/base.py` | `Detector` interface, `DetectorResult`, `DetectorInfo`, `WeightsSpec`, `AnalysisContext`, typed errors |
| `backend/app/detectors/registry.py` | Config-driven selection: `get(kind)`, `settings(kind)`, `model_dir()`, `warm_up()`, `override()` for tests |
| `backend/app/detectors/testing.py` | `StubDetector`, `TinyTorchDetector` for tests and tooling smoke checks |
| `backend/app/detectors/__init__.py` | Re-exports |
| `backend/config.yaml` | `detectors.<kind>.{enabled, model, weight, options}`, `pipeline.video.*` |
| `README.md`, `docs/model-choice.md`, `docs/upgrade-notes.md`, this file | Written by the lead after merging |

A detector package registers implementations by exporting
`DETECTORS: dict[str, type[Detector]]` and `DEFAULT: str | None` from its
`__init__.py`. Config `model: default` resolves to that `DEFAULT`. Detector-specific
settings come from `ctx.options` / `self.options` (config `detectors.<kind>.options`)
with defaults in code — document your keys; the lead adds them to `config.yaml`.

## Ownership

| Workstream | Branch | Owns (create/edit/delete only these) |
|---|---|---|
| **A — image** | `agent-a/image-model` | `backend/app/detectors/image/**`, `backend/app/image_detection.py`, `backend/tests/test_detectors_image.py`, `backend/scripts/check_image_models.py`, `docs/model-choice-image.md`, `requirements/extra-agent-a.txt` |
| **B — audio** | `agent-b/audio-model` | `backend/app/detectors/audio/**`, `backend/app/audio_detection.py`, `backend/tests/test_detectors_audio.py`, `backend/scripts/check_audio_models.py`, `docs/model-choice-audio.md`, `requirements/extra-agent-b.txt` |
| **F — video** | `agent-f/video-model` | `backend/app/detectors/video/**`, `backend/app/pipeline.py`, `backend/tests/test_detectors_video.py`, `backend/tests/test_pipeline.py`, `backend/scripts/check_video_models.py`, `docs/model-choice-video.md`, `requirements/extra-agent-f.txt` |
| **C — product/API** | `agent-c/product` | `backend/app/main.py`, `backend/app/service.py`, `backend/app/config.py`, `backend/app/ai_summary.py`, new `backend/app/{auth,ratelimit,jobs,validation}.py`, `backend/templates/**`, `backend/.env.example`, `backend/tests/test_service.py`, `backend/tests/test_api.py`, `backend/requirements_working.txt` (delete), `requirements/extra-agent-c.txt` |
| **D — packaging/deploy** | `agent-d/deploy` | `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.gitignore`, `backend/.gitignore`, `.gitattributes`, `pyproject.toml`, `backend/pyproject.toml`, `requirements.txt`, `backend/requirements.txt`, `backend/uv.lock`, any new lock file, `backend/.python-version`, `.github/**`, `backend/scripts/download_models.py`, `backend/models/**` (untracking), `deploy/**`, `docs/lfs-cleanup.md`, `docs/deploy-hf-spaces.md` |
| **E — MLOps** | `agent-e/mlops` | `backend/scripts/evaluate_models.py`, `backend/scripts/export_onnx.py`, `backend/scripts/quantize_onnx.py`, `backend/scripts/benchmark_models.py`, `backend/mlops/**`, `backend/tests/test_mlops.py`, `docs/mlops.md`, `requirements/extra-agent-e.txt` |

Not owned by anyone during the parallel phase (do not edit): `backend/test/**`
(sample media), `backend/notebook/**`, `resources/**`, `LICENSE`.

## Dependencies

D owns the dependency manifest. Any other workstream that needs a new package
installs it into the shared venv and lists it, with the version it tested, in its
own `requirements/extra-agent-<x>.txt`. The lead folds those into D's pinned
manifest at merge time.

## Merge order

A → B → F → C → D → E into `upgrade/v2`. Nothing is merged into `main`.
