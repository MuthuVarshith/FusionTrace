# Video temporal signal — model choice (workstream F)

**Result: no learned model passed. DEFAULT is `frame-consistency-heuristic`, a
non-learned, uncalibrated heuristic** (`backend/app/detectors/video/`).

## Selection criteria

To be considered, a candidate had to model time across frames, use a permissive
licence (MIT, Apache-2.0, BSD or CC-BY), ship safetensors weights (or pickle
weights that load with `weights_only=True`), run without `trust_remote_code`,
run on CPU, and be about 1 GB or smaller. Hub searched 2026-10-11 (model
search terms: deepfake video, videomae/timesformer deepfake, lip sync, syncnet,
lipforensics, faceforensics, dfdc, celeb-df, fakeavceleb, lav-df, av-hubert,
genconvit, temporal/lstm/3d variants; plus `video-classification` pipeline tag).

| Candidate (revision) | Licence | Size | Architecture / training data | Decision |
|---|---|---|---|---|
| Vansh180/VideoMae-ffc23-deepfake-detector (`ea26518`) | MIT label | 345 MB safetensors | VideoMAE-B, 16 frames, FF++ c23 (Deepfakes subset); self-reported AUC 0.836 | **Rejected**: fine-tuned from MCG-NJU/videomae-base-finetuned-kinetics, which is **CC-BY-NC-4.0**; the MIT label cannot relax that. Technically the best fit otherwise. |
| eftt/VideoMae-ffc23-deepfake-detector (`b18f2a0`) | MIT label | 345 MB | Copy of the above | Rejected (same reason) |
| shylhy/videomae-large-finetuned-deepfake-subset, Hemgg/* VideoMAE, gautamtata/videomae-*lipsync*, muneeb1812/videomae-* | cc-by-nc-4.0 or none | 0.3–1.2 GB | VideoMAE fine-tunes, small undocumented subsets | Rejected: non-commercial / no licence |
| mhamza-007/timesformer-deepfake-detection (`c78fc30`) | MIT | no weights | — | Rejected: empty repo; TimeSformer base is CC-BY-NC-4.0 anyway |
| ash12321/deepfake-detector-timesformer | none | pickle ckpt | One-class TimeSformer, WebVid; tested only on "ultra-extreme" fakes | Rejected: no licence, pickle, unvalidated |
| Naman712/Deep-fake-detection (`103847a`) | MIT | 227 MB `.pt` | ResNeXt50 + LSTM, 20 frames, "custom dataset", 87% internal acc. | Rejected: gated (requires accepting terms), pickle, undocumented training data |
| maddy08/deepfake-video-detection (`b6b2c77`) | Apache-2.0 | 227 MB `.pt` | Undocumented (empty card; same size as the above) | Rejected: no architecture/training documentation |
| KoreaPeter/ms-eff-gcvit-deepfake-b0/b5-celeb-df-v2 (`7799a86` / `654a1b5`) | MIT | 37 / 220 MB | EfficientNet+GCViT, Celeb-DF v2, **frame-level** + mean aggregation | Rejected: not temporal, needs `trust_remote_code`, YOLOv8 face detector pickle (AGPL upstream); cross-dataset AUC ~0.73 per card |
| oo01/deepfake-video-classifier, razaali313/deepfake-video-classifier (`ebdeb16`, `31ba508`) | MIT | 52 MB `.pth` | EfficientNet-B0 + temporal **mean pooling** (order-invariant), Celeb-DF v2 | Rejected: not temporal, pickle |
| HoopitAI/video-deepfake-detection-GenD_* | MIT | — | CLIP/DINOv3 per-frame | Rejected: per-frame image models (workstream A territory) |
| Deressa/GenConViT, Thanuja2109/GenConViT | cc-by-nc-4.0 / MIT copy | — | Per-frame ConvNeXt+Swin AE/VAE | Rejected: frame-level; upstream NC |
| anushavyas/deepfake-detector-vision-kan-lstm (`ab5e965`) | MIT | 74 MB `.pt` | Vision-KAN + Vision-LSTM over image patches (single image) | Rejected: not temporal, `torch.hub.load` remote code |
| AUA27/deepfake-multimodal-weights (`41bbedb`) | MIT (video part keeps upstream terms) | 372 MB | SigLIP per-frame + sklearn pickle fusion | Rejected: per-frame, pickle fusion |
| ckoutlis/auvire-lavdf, ckoutlis/auvire-avdeepfake1m (`d7677c9`, `d5099a6`) | Apache-2.0 | 48 / 36 MB | AuViRe audio-visual temporal localisation head | Rejected: head only; needs AV-HuBERT features (non-commercial upstream licence, fairseq, lip ROI pipeline), no model code on Hub; trained on NC datasets |
| advy/fakeavceleb-detect | Apache-2.0 | 250 MB+ per `.pt` | Audio + visual branches, FakeAVCeleb | Rejected: pickle, undocumented, repo is training dump |
| torrwill/lipsyncnet-checkpoints | MIT | 0.8–3.7 GB `.pt` | Lip reading on GRID | Rejected: not a detector, size |
| braindecode/SyncNet, lithiumice/syncnet, nawta/syncnet | BSD / none / other | — | EEG model / unlicensed | Rejected: wrong task or no licence |

**What would change this:** a VideoMAE/TimeSformer-style detector fine-tuned
from a permissively licensed backbone (e.g. X-CLIP, MIT) with documented
training data. The integration points (clip sampling in `sampling.py`, budget
handling, export hooks per clip) are already in place.

## The heuristic (DEFAULT)

`frame-consistency-heuristic` samples `max_clips` (4) clips of
`frames_per_clip` (8) consecutive frames evenly across the video and computes:

* **flicker** — motion-compensated temporal roughness of 16 px block luminance
  (second difference over central first difference + 1 grey level), 95th
  percentile over blocks, median over clips. Formula in
  `backend/app/detectors/video/consistency.py`.
* **score_jitter** — mean |risk(t+1) − risk(t)| of the configured image
  detector over 4 consecutive frames per clip.

`risk = weighted mean(0.6 · ramp(jitter, 0.02→0.20), 0.4 · ramp(flicker, 1.0→3.0))`
over the available components. Ramps were chosen a priori and **not tuned** on
any data. Options: `max_clips`, `frames_per_clip`, `max_seconds` (120),
`score_jitter` (true), `score_frames_per_clip` (4), `threshold` (0.5).

## Quick checks actually run

* Synthetic ffmpeg clips (sample photo with slow zoom): smooth clip risk 0.000
  (flicker 0.66); same clip with a box blinking every other frame risk 1.000
  (flicker 32.7). Proves the code path and the flicker measurement work, nothing
  about real deepfakes.
* Discrimination sanity check, flicker component only (`--no-image`; no image
  detector exists in this branch): dataset `angads24/deepfake-video` (revision
  `3470af66b50e6d715f108bdfa67342907137e1f4`, a re-upload of Google's DFD
  face-swap set, licence label cc-by-sa-4.0), **n = 10 real + 10 fake**, scenes
  `exit_phone_room` and `walk_down_hall_angry`, 1080p, ~10 s each.
  `scripts/check_video_models.py` result: every clip risk 0.000 (raw flicker
  0.15–0.52), so **risk AUC 0.50**. Raw flicker was somewhat higher on fakes
  (median ≈0.30 vs ≈0.23; raw-value AUC 0.77, computed by hand from the printed
  values) but stays far below the ramp start, i.e. this face-swap set does not
  show the gross flicker the heuristic is built to catch. ~2.5–3.5 s per clip on
  CPU. Twenty clips from one dataset prove almost nothing; the thresholds were
  deliberately not refitted to them.
* Score jitter was only checked with stub image detectors (unit tests).

**Bottom line:** the heuristic will flag gross temporal artefacts and otherwise
contributes a low risk. Because the pipeline uses a weighted mean, a low
temporal risk pulls a video's score down; until calibrated, a lower
`detectors.video.weight` (e.g. 0.3) is advisable.
