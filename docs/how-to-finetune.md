# How to fine-tune the image detector (LoRA)

This fine-tunes the default image detector, `commfor-vit-s384` (Community Forensics ViT-S/16-384),
with LoRA adapters on the attention layers. The result is an **adapter**, not a new base model.
It is logged to MLflow and registered with the alias **`candidate`**. Nothing is promoted to
`production` automatically.

> **Status.** The pipeline is built and smoke-tested end to end (3 training steps, on the local test
> images). **No full training has been run and no fine-tuned model exists yet.** Two inputs are still
> missing, see [Blocked inputs](#blocked-inputs).

## What the pipeline guarantees

| Rule | How it is enforced |
|---|---|
| Groups never straddle splits | `finetune/data.py: check_disjoint`: training refuses a manifest where a group (identity or source video) appears in two splits |
| Test data is never trained on or thresholded | Training reads only `train` and `val` rows; `evaluate_finetune.py` never writes a threshold back |
| Threshold is chosen on validation only | `scripts/finetune_lora.py: pick_threshold` maximises balanced accuracy on `val` |
| Best epoch is chosen on validation only | Highest `val` AUC |
| Data version is recorded | SHA-256 of the manifest contents, logged with every run |
| Never auto-promoted | MLflow alias `candidate` only; `production` is a separate, deliberate step |

## Data

**Record of datasets reviewed (2026-10-11):**

| Dataset | Licence | Content | Use here |
|---|---|---|---|
| `bitmind/face-swap` | Apache-2.0 | face-swap images: `train` 8,076, `validation` 1,010, `test` 1,010 (parquet). **No real images. No identity or source-video metadata.** | Training fakes (see the leakage warning below) |
| `skyhawk000/semisynth-faceswap` | Apache-2.0 | **Identical files** to `bitmind/face-swap` (same filenames and sizes). Not an independent second dataset. | Do not count as a second test set |
| `LucaCorvitto/DeepFake-Faces-Stimuli` | MIT | Stable Diffusion faces (`realistic-vision-v51`), which **is in Community Forensics' training metadata** | Regression set (b) |
| FairFace (real faces, used in the v2 quick check) | see `docs/model-choice-image.md` | real faces | Negatives for the regression check |
| FFHQ, CelebA | **non-commercial** | real faces | Not used without your decision (see below) |

**Leakage warning for face swaps.** Filenames are numbered (`hq_swap_1234.jpg`,
`swap_56_to_celeb.jpg`) with no identity or source-video field, so the split cannot be made
identity-disjoint from the data alone. Until identity or video IDs are available, the
`--group-by stem` split treats every image as independent, which **probably leaks**. The
validation numbers from such a split are optimistic. The script prints a warning when you
use it.

## Blocked inputs

These need a decision from you before a real run:

1. **Real training faces.** The common real-face sets (FFHQ, CelebA) are non-commercial. A
   permissively licensed set was not found. Options: (a) accept a research-only licence for
   training; (b) a set you provide; (c) a set we find later. Put them in `data/train/real/`
   (or any folder you name in the manifest).
2. **A second face-swap method for test set (a).** `skyhawk000/semisynth-faceswap` is a
   duplicate. A different, licence-checked face-swap test set is needed, e.g. a set with
   FaceForensics++-style methods (FF++ has its own non-commercial terms).

## Commands

Run from `backend/`. Use the shared venv on Windows (`..\.venv\Scripts\python.exe`), or a
virtualenv elsewhere.

### 1. Install

```bash
pip install -r ../requirements.txt
pip install peft==0.21.2 mlflow-skinny==3.17.0
```

### 2. Build the manifest

Folder layout:

```
data/train/real/...   data/train/fake/...
```

With identity or source-video subfolders (preferred, e.g. `data/train/fake/<video_id>/...`):

```bash
python scripts/make_manifest.py --real data/train/real --fake data/train/fake \
    --out data/manifest.csv --fractions 0.8,0.1,0.1 --group-by parent
```

Flat folders (no identity info, only if the images are known to be independent):

```bash
python scripts/make_manifest.py --real data/train/real --fake data/train/fake \
    --out data/manifest.csv --fractions 0.8,0.1,0.1 --group-by stem
```

The script prints the rows per split and the `data_version`. It fails if a group appears in two splits.

### 3. Smoke test (a few minutes on CPU)

```bash
python scripts/finetune_lora.py --manifest data/manifest.csv --out runs/smoke \
    --epochs 1 --batch-size 4 --workers 0 --max-train-steps 3
```

### 4. Full training, CPU

Slow: a ViT-S at 384 px with batch 16 takes several seconds per step on a 12-thread CPU.

```bash
python scripts/finetune_lora.py --manifest data/manifest.csv --out runs/lora-1 \
    --epochs 3 --batch-size 16 --lr 1e-4 --rank 8 --alpha 16 --device cpu --mlflow
```

### 5. Full training, Google Colab (free GPU)

In Colab: **Runtime → Change runtime type → T4 GPU**. Then, in cells:

```bash
!git clone -b upgrade/v2 https://github.com/MuthuVarshith/FusionTrace.git
%cd FusionTrace
!pip install -q -r requirements.txt peft==0.21.2 mlflow-skinny==3.17.0
%cd backend
!python scripts/download_models.py          # fetches the default commfor weights from the Hub
```

Put the images in `data/train/...` (upload, or copy from Google Drive: `from google.colab import drive`),
then build the manifest and train:

```bash
!python scripts/make_manifest.py --real data/train/real --fake data/train/fake --out data/manifest.csv --group-by parent
!python scripts/finetune_lora.py --manifest data/manifest.csv --out runs/lora-1 --epochs 3 --batch-size 16 --device cuda --mlflow
```

Copy `runs/lora-1` to Drive before the session ends (`!cp -r runs/lora-1 /content/drive/MyDrive/`).

**MLflow on Colab** uses a local file store by default. To keep runs, point
`MLFLOW_TRACKING_URI` at a folder in Drive, or copy `../.mlruns` out afterwards.

### 6. Evaluate base vs fine-tuned

Test sets are folders with `real/` and `fake/` subfolders. Use **test sets that were not in the
manifest**:

```bash
python scripts/evaluate_finetune.py --adapter runs/lora-1 \
    --set faceswap_test=data/test_faceswap/real:data/test_faceswap/fake \
    --set ai_faces=data/test_public/real:data/test_public/fake \
    --out runs/lora-1/eval.json
```

Output is a Markdown table with AUC, fake recall and real recall for each test set, for the base
model (threshold 0.5) and the fine-tune (threshold from `threshold.json`).

### 7. Promotion (manual, not part of this pipeline)

Only after reviewing the evaluation:

```bash
python -c "from mlops import model_registry as r; print(r.promote('fusiontrace-image-lora', 'candidate', 'production'))"
```

The app does not yet load LoRA adapters. Wiring an adapter into the detector registry
(`app/detectors/image/`) is a separate step; the registry entry will read `threshold.json`
for its threshold.

## Outputs

```
runs/lora-1/
  adapter/          PEFT adapter (LoRA weights + trained head)
  threshold.json    threshold, validation metrics, best epoch, data_version, args, history
  manifest.csv      copy of the manifest used
  eval.json         (after evaluate_finetune.py) per-set metrics for base and fine-tune
```

## Known limits

- The LoRA target is the attention `qkv` and `proj` layers of the ViT (`221,569` trainable
  parameters, about 1% of the model, including the head).
- Augmentation is a random crop and horizontal flip only.
- Face-swap identity leakage is unverified until identity or video IDs are available.
- The base model's diffusion-face AUC was 1.000, but its training data includes that generator, so
  the regression set (b) is not an unseen-generator test. The face-swap result (AUC 0.459 before
  fine-tuning) is the main signal of improvement.
