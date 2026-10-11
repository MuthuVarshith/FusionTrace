# Image detector choice (workstream A, 2026-10-11)

**Default: `commfor-vit-s384`.** Community Forensics ViT-S/16-384, from `OwensLab/commfor-model-384`. It is also the smallest candidate (87 MB).

## Candidates (Hugging Face Hub, searched 2026-10-11)

| Registry key | Repo @ revision | Licence | Size | Architecture | Training data |
|---|---|---|---|---|---|
| `commfor-vit-s384` | `OwensLab/commfor-model-384` @ `6076002b` | MIT | 87 MB safetensors | timm `vit_small_patch16_384`, 1 logit (sigmoid = P(fake)); resize 440 → centre-crop 384 → ImageNet normalisation, as in the official `dataloader.get_transform(mode="test")` | Community Forensics: 2.7M images from 4,803 generators (Park & Owens, CVPR 2025) |
| `gend-clip-l14` (CLIP-based, UnivFD-style) | `yermandy/GenD_CLIP_L_14` @ `891ce014` | MIT | **1.22 GB, which misses the ~1 GB limit** | CLIP ViT-L/14 with only the LayerNorms tuned; the L2-normalised CLS feature goes to a linear probe with 2 classes (index 1 = fake, as in the repo's `app/run.py`) | FaceForensics++ face crops (RetinaFace, scale 1.3). WACV 2026 |
| `siglip-deepfake-v1` | `prithivMLmods/deepfake-detector-model-v1` @ `c5cb24c6` | Apache-2.0 | 372 MB | SigLIP base/16 @224 with a linear head on mean-pooled tokens; class 0 = Fake | `prithivMLmods/OpenDeepfake-Preview`. The author marks it "experimental" |
| `efficientnetv2-legacy` | local / private mirror (`HF_REPO_ID`) | MIT (project) | 470 MB .pth | `tf_efficientnetv2_l` with a custom head | Kaggle 140k-real-and-fake-faces + real-vs-ai-generated-faces |

All candidates load safetensors with stock `timm`/`transformers` classes. No `trust_remote_code` is used, and no repo code is executed. GenD's `modeling_gend.py` was read only to rebuild the architecture.

Rejected candidates:
- `umm-maybe/AI-image-detector`: trained on art, not faces.
- `Smogy/*`: CC-BY-NC.
- `buildborderless/CommunityForensics-DeepfakeDet-ViT`: a third-party conversion of the same weights, with a history of wrong weights and CLIP normalisation instead of the official ImageNet normalisation.
- `Bombek1/*`: ships pickle weights only.
- No UnivFD linear-probe weights are hosted on the Hub. GenD is the closest Hub-hosted equivalent.

## Quick check (`backend/scripts/check_image_models.py`)

Both sets were run on CPU with 2 threads. The decision threshold is 0.5. Each model ran in its own process.

- **local**: `backend/test`, 9 fake and 11 real images. This set is very likely drawn from the legacy model's training distribution: the file names match the 140k-faces and real-vs-ai-faces Kaggle sets.
- **public**: an out-of-distribution sample of 90 fake and 90 real images.
  - Real: `HuggingFaceM4/FairFace` @ `54d573cd`, config 1.25, validation split, streaming `shuffle(seed=0, buffer_size=1000).take(90)`. Licence CC-BY-4.0.
  - Fake: `LucaCorvitto/DeepFake-Faces-Stimuli` @ `28bd8dc6`, 90 files sampled with seed 0. These are Stable Diffusion Realistic Vision 5.1 faces. Licence MIT.
  - Both classes were processed identically: centre square crop, 256×256, JPEG quality 90.
  - Stored in `C:/FusionTrace/.data/agent-a/public` with a `manifest.json`.

| Model | Set | n fake/real | Fake recall | Real recall | AUC | s/img |
|---|---|---|---|---|---|---|
| commfor-vit-s384 | local | 9/11 | 1.000 | 0.909 | 0.919 | 0.77 |
| commfor-vit-s384 | public | 90/90 | **0.989** | **0.989** | **1.000** | 0.71 |
| gend-clip-l14 | local | 9/11 | 0.333 | 0.909 | 0.798 | 2.81 |
| gend-clip-l14 | public | 90/90 | 0.133 | 0.956 | 0.700 | 2.09 |
| siglip-deepfake-v1 | local | 9/11 | 0.444 | 0.818 | 0.788 | 3.48 |
| siglip-deepfake-v1 | public | 90/90 | 0.100 | 0.467 | 0.155 | 1.28 |
| efficientnetv2-legacy (ImageNet norm) | local | 9/11 | 0.889 | 0.909 | 0.970 | 1.23 |
| efficientnetv2-legacy (ImageNet norm) | public | 90/90 | 0.011 | 1.000 | 0.821 | 1.02 |
HALFNORM_ROWS

The s/img column includes first-call warm-up and the load of other agents' processes on the machine. Treat it as a rough figure.

**Decision rule.** A model wins on public-set AUC. If two models are within 0.03 AUC of each other, the smaller one wins.

CommFor is far ahead and is also the smallest, so the rule is not needed here. The legacy model reproduces the reported failure: on out-of-distribution fakes it answers Real almost every time (fake recall 0.011), even though its AUC is 0.82. That combination suggests a badly placed threshold on top of weak features.

**Caveats:**
- The sample is small (n = 180) and contains a single generator family. It has no face swaps.
- The fakes come from one source and the reals from another. The shared processing removes format and resolution cues, but scene and style differences between the two sources remain.
- GenD was trained on face-swap crops, and FusionTrace bundles no face detector, so it is disadvantaged here.
- Community Forensics may have seen Stable Diffusion-family generators during training.

These numbers are not a general accuracy claim.

## Legacy preprocessing check (v1 `app/image_detection.py` vs `backend/notebook/image_detection/`)

- **Normalisation mismatch (a real bug).** Training used ImageNet mean/std: `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` in `image_deepfake_detection (2).ipynb` cell 2 and in `model_training.py` around line 310. The v1 inference code used mean = std = 0.5. The wrapper now defaults to `normalization: imagenet`; `half` keeps the v1 behaviour for comparison.
- **Class order is consistent.** Training labels a folder named `real` as 0 and everything else as 1 (notebook cell 1). Inference treats sigmoid ≥ 0.5 as Fake, which matches.
- **RGB and resize are consistent.** Both use PIL RGB and `Resize((224, 224))`. Training adds random crop, flip and jitter augmentations.
- **No face cropping at either stage.** Training images are already-aligned face crops (FFHQ/StyleGAN at 256 px). Arbitrary user photos are therefore far outside the training distribution.
- **Threshold is 0.5 in both.**
- **Variant training script.** `model_training.py` (around line 410) appends `nn.Sigmoid()` to the head and also uses `BCEWithLogitsLoss`, which applies a sigmoid twice. The notebook that logged the 0.99 validation accuracy does not have this. The deployed weights cannot tell us which script produced them, because the Sigmoid layer has no parameters.
- **Training data overlap.** The local `backend/test` images come from the training datasets, so their 95%+ scores say nothing about generalisation.

## Explanations and concurrency

`explain()` writes a JPEG overlay using a jet colormap built in NumPy and PIL (no OpenCV):
- **ViTs:** GradCAM over the patch tokens of the residual stream entering the last block, reshaped to the patch grid. CommFor uses a 24×24 grid.
- **Legacy CNN:** GradCAM on `blocks[-1]`, as in v1.

The forward pass is staged inside the call, so no hooks are registered and `.backward()` is never called:
1. The trunk runs under `no_grad`.
2. A detached copy of the activation is created with `requires_grad=True`.
3. `torch.autograd.grad` is computed with respect to that copy.

Parameters are frozen and in eval mode from load onward, so no gradients or module state are ever shared between calls. A per-instance `_explain_lock` only bounds memory.

## Options (`detectors.image.options`)

- `threshold`: float, default 0.5. Applies to all image detectors.
- `normalization`: `imagenet` (default) or `half`. Legacy model only.
