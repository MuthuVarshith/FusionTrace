# Model choice — FusionTrace v2

One page per signal, with every candidate considered, licences, revisions, the datasets and
the raw quick-check results:

- [Image](model-choice-image.md)
- [Audio](model-choice-audio.md)
- [Video (temporal)](model-choice-video.md)

This page is the summary. All numbers come from small checks run once on CPU during the
upgrade (2026-10-11). They show the models behave plausibly; they are not benchmarks.

## Selection rules (all signals)

- Licence must be permissive: MIT, Apache-2.0, BSD or CC-BY. Non-commercial and
  research-only licences are excluded, including ones inherited from a base model.
- Weights must be safetensors, or pickle that loads with `weights_only=True`.
- No `trust_remote_code`, and no repo code executed.
- About 1 GB of weights or less, and runnable on CPU.
- Revision pinned in the detector's `WeightsSpec`.

## Defaults

| Signal | Default | Why | Key quick-check result |
|---|---|---|---|
| Image | `commfor-vit-s384`: Community Forensics ViT-S/16-384 (`OwensLab/commfor-model-384` @ `6076002b`, MIT, 87 MB) | Best AUC on the public faces, and the smallest candidate. **Caveat:** that sample's generator is in its training data, and it fails on face swaps | AUC 1.000 on diffusion faces (generator seen in training); **AUC 0.459 on 90 face swaps** (unseen type) |
| Audio | `wav2vec2-legacy` (v1 Wav2Vec2-base, private weights) | Both Hub candidates were clearly worse | AUC 0.916 on 50 + 50 In-the-Wild clips |
| Video, temporal | `frame-consistency-heuristic` | No learned temporal model met the licence and safety rules | AUC 0.50 on 10 + 10 public clips: it does not separate them, so its weight is 0 (computed, not scored) |

## Image: public out-of-distribution sample (90 real FairFace + 90 Stable Diffusion faces)

| Model | Size | Fake recall | Real recall | AUC | Local AUC (9 fake + 11 real) |
|---|---|---|---|---|---|
| `commfor-vit-s384` | 87 MB | 0.989 | 0.989 | 1.000 | 0.919 |
| `gend-clip-l14` (CLIP ViT-L/14, UnivFD-style) | 1.22 GB | 0.133 | 0.956 | 0.700 | 0.798 |
| `siglip-deepfake-v1` | 372 MB | 0.100 | 0.467 | 0.155 | 0.788 |
| `efficientnetv2-legacy`, ImageNet normalisation (as trained) | 470 MB | 0.011 | 1.000 | 0.821 | 0.970 |
| `efficientnetv2-legacy`, v1 inference normalisation | 470 MB | 0.000 | 1.000 | 0.791 | 1.000 |

**Overlap with training data.** `realistic-vision-v51`, the generator behind the public fake set
(`LucaCorvitto/DeepFake-Faces-Stimuli`), is listed in Community Forensics' generator metadata
(`data/metadata/systematic_set_metadata.csv`). So the 1.000 above measures recognition of a
generator seen in training. The real faces (FairFace) are not in its real-data mix (CelebA, FFHQ,
MetFaces, LAION, ImageNet).

## Image: unseen fakes, face swaps (`bitmind/face-swap` test split, Apache-2.0)

90 face-swap images against the same 90 FairFace real faces. Community Forensics covers image
generators and has no face-swap tools, so these fakes are unseen by construction.

| Model | Fake recall | Real recall | AUC |
|---|---|---|---|
| `commfor-vit-s384` | 0.011 | 0.989 | **0.459** |

The default image model is at chance on face swaps. Its diffusion-face results do not transfer
to this fake type.

## Image: threshold check, `efficientnetv2-legacy` (v1)

On the public faces the v1 model has AUC 0.821 but catches 1.1% of fakes at 0.5, because its
fake scores sit near 0 (median 0.0005). At a threshold of about 0.001 it catches 42% of fakes at
92% real recall. That threshold was read off the same sample, so it is fitted to it. The v2
default keeps 0.5.

## Label mapping check (siglip, melodymachine)

Both mappings match their configs and model cards (siglip: 0 = Fake; melodymachine: 0 = fake),
and their preprocessing matches the configs. Their low AUCs are not a label swap: a flip would
improve one dataset and hurt the other. Both fail to transfer to these datasets.

The v1 model scores well on its own test folder and labels almost every out-of-distribution
fake as real. That is the generalization failure that motivated v2. A preprocessing bug
contributed: v1 normalized with mean/std 0.5 at inference while training used ImageNet
statistics. Fixing it does not restore generalization.

`gend-clip-l14` exceeds the ~1 GB size limit. It is kept as an option because it is the only
CLIP-based candidate, and no UnivFD linear-probe weights are hosted on the Hub. It expects
face crops (it was trained on RetinaFace crops), and FusionTrace has no face detector, which
likely hurts it.

**Limitation:** the public fakes are diffusion-generated faces, close to Community Forensics'
training distribution. Face swaps, face reenactment and edited real photos were not tested.

## Audio: In-the-Wild sample (50 real + 50 fake)

| Model | Real recall | Fake recall | AUC |
|---|---|---|---|
| `wav2vec2-legacy` | 1.000 | 0.740 | 0.916 |
| `wav2vec2-xlsr-garystafford` | 0.900 | 0.440 | 0.682 |
| `wav2vec2-base-melodymachine` | 0.440 | 0.180 | 0.296 |

## Video temporal signal

About 20 Hub candidates were reviewed. The best technical fit was a VideoMAE-B model trained
on FaceForensics++ (`Vansh180/VideoMae-ffc23-deepfake-detector`). It was rejected because it is
fine-tuned from a CC-BY-NC-4.0 base model. The rest were per-frame only, pickle-only, gated,
needed remote code, or were unlicensed.

The default heuristic samples up to 4 clips of 8 consecutive frames. It measures pixel flicker
between frames and jitter in the image model's scores across them, and maps both to risk with
an uncalibrated formula.

On `angads24/deepfake-video` @ `3470af6` (10 real + 10 fake, flicker part only):

| Measure | AUC |
|---|---|
| Final risk | 0.50: every clip scored 0 |
| Raw flicker statistic | 0.77 |

So the raw measure leans toward fakes, but the scaling maps everything below its starting
point to zero. Recalibrating that scaling is the obvious next step.
