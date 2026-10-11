# Audio detector choice (workstream B, check run by the lead, 2026-10-11)

**Decision:** `wav2vec2-legacy` stays the default. Neither Hub candidate came close to it.

## Candidates

| Registry key | Repo @ revision | Licence | Size | Architecture / training |
|---|---|---|---|---|
| `wav2vec2-legacy` (default) | local / private mirror (`HF_REPO_ID`, `audio_model/*`) | MIT (project) | 361 MB safetensors | Wav2Vec2-base sequence classifier, 4 s clips at 16 kHz; class 0 = Real, 1 = Fake |
| `wav2vec2-xlsr-garystafford` | `garystafford/wav2vec2-deepfake-voice-detector` @ `c663060` | Apache-2.0 | 1.26 GB | Wav2Vec2 XLS-R 300M classifier |
| `wav2vec2-base-melodymachine` | `MelodyMachine/Deepfake-audio-detection-V2` @ `de3cde5` | Apache-2.0 | ~360 MB | Wav2Vec2-base classifier |

All three load safetensors through stock `transformers` classes. None uses `trust_remote_code`.

## Quick check (`backend/scripts/check_audio_models.py`)

How the check was run:
- one model per process, CPU, threshold 0.5
- aggregate `top_k_mean` with k = 3 over 4 s windows with a 2 s hop

Datasets:
- `local`: `backend/test/{real,deepfakes}/*.wav` (9 real, 6 fake)
- `inthewild`: 50 real + 50 fake clips from `SpeechAntiSpoofingBenchmarks/InTheWild` @ `f36fbe7e`, saved under `.data/agent-b/inthewild` and not committed

| Model | Dataset | n real / fake | Real recall | Fake recall | AUC |
|---|---|---|---|---|---|
| `wav2vec2-legacy` | local | 9 / 6 | 0.667 | 1.000 | 0.907 |
| `wav2vec2-legacy` | inthewild | 50 / 50 | 1.000 | 0.740 | **0.916** |
| `wav2vec2-xlsr-garystafford` | local | 9 / 6 | 0.778 | 0.000 | 0.537 |
| `wav2vec2-xlsr-garystafford` | inthewild | 50 / 50 | 0.900 | 0.440 | 0.682 |
| `wav2vec2-base-melodymachine` | local | 9 / 6 | 1.000 | 0.333 | 0.648 |
| `wav2vec2-base-melodymachine` | inthewild | 50 / 50 | 0.440 | 0.180 | 0.296 |

CPU cost per 4 s window, with 4 threads:

| Model | Seconds per window |
|---|---|
| legacy | 1.43 |
| XLS-R | 1.61 |
| MelodyMachine | 0.66 |

**Switching rule (set before the run):** a candidate replaces the default only if it beats the
default on the In-the-Wild sample by at least 0.10 AUC and 0.10 balanced accuracy, and loses no
more than 0.10 balanced accuracy on the local WAVs. Neither candidate meets it; both are worse
than the default on both datasets.

Caveat: n = 100 is a small sample, and In-the-Wild is one corpus. The legacy model doing well
here does not mean it generalizes to every voice-cloning system.

## Whole-clip windowing and the aggregate

v1 trimmed every clip to its first 4 s. v2 scores the whole clip in sliding windows:

| Option (`detectors.audio.options`) | Default | Meaning |
|---|---|---|
| `window_seconds` | 4.0 | Window length; the legacy model was trained on 4 s clips |
| `hop_seconds` | 2.0 | Step between windows |
| `max_windows` | 60 | Windows are spread evenly across the clip when the cap is hit |
| `aggregate` | `top_k_mean` | `mean`, `max` or `top_k_mean` |
| `top_k` | 3 | Windows averaged by `top_k_mean` |
| `threshold` | 0.5 | Fake/Real label threshold |
| `silence_dbfs` | -60 | Quieter windows are skipped |
| `batch_size` | 1 | Windows per forward pass |

Clips shorter than one window are padded. Empty, silent, or shorter than 0.25 s clips raise
`UnsupportedInputError`. The report lists every window's score with its start and end time,
plus the mean, the max, and the aggregate used.

**Why `top_k_mean`.** The splice test inserts one fake clip into an otherwise real recording:
10 real-only recordings vs 10 spliced, from In-the-Wild, scored by the legacy model.

| Aggregate | Real recall | Fake recall | AUC |
|---|---|---|---|
| `mean` | 1.0 | **0.0** | 0.830 |
| `max` | 0.8 | 0.8 | 0.850 |
| `top_k_mean` (k = 3) | 0.9 | 0.8 | **0.885** |

`mean` dilutes a short synthetic segment until it disappears. `max` catches it but flags more
real recordings, because a single noisy window is enough. `top_k_mean` caught as many splices
as `max` while flagging fewer real recordings. The sample is small (n = 20), so treat this as
a sensible default, not a tuned one.
