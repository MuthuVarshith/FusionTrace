# How to fine-tune the image detector (LoRA)

This fine-tunes the default image detector, `commfor-vit-s384` (Community Forensics ViT-S/16-384),
with LoRA adapters on the attention layers. The output is an **adapter**, not a new base model.
It is logged to MLflow and registered with the alias **`candidate`**. Nothing is promoted to
`production` automatically.

> **Status.** The pipeline is built and tested, and the dataset is built (`.data/finetune/`, not
> in git). **No full training has been run, so no fine-tuned model exists yet.** A smoke run (3
> steps on local test images) and the adapter loading path were verified end to end. See
> [Known limits](#known-limits).

## What the pipeline guarantees

| Rule | How it is enforced |
|---|---|
| Groups never straddle splits | `finetune/data.py: check_disjoint`: training refuses a manifest where a group appears in two splits |
| Test data is never trained on or thresholded | Training reads only `train` and `val` rows; `evaluate_finetune.py` never writes a threshold back |
| Threshold is chosen on validation only | `scripts/finetune_lora.py: pick_threshold` maximises balanced accuracy on `val` |
| Best epoch is chosen on validation only | Highest `val` AUC |
| Data version is recorded | SHA-256 of the manifest contents, logged with every run and stored on the registry version |
| Face-swap results are labelled | Reports carry `same-source only` on every face-swap set until a second licence-checked set is used |
| Never auto-promoted | MLflow alias `candidate` only; `production` is a separate, deliberate step |

## Data

### Sources and licences

| Source | Used for | Licence | Attribution |
|---|---|---|---|
| `HuggingFaceM4/FairFace` (validation parquet, 0.25 padding), 5,000 sampled | **real faces** | CC BY 4.0 | Kärkkäinen & Joo, *FairFace*, WACV 2021 |
| `bitmind/face-swap` (train, validation and test parquet), all 10,096 images | **face-swap fakes** | Apache-2.0 | bitmind |
| `LucaCorvitto/DeepFake-Faces-Stimuli`, 90 images | regression fakes (not trained on) | MIT | — |

`.data/finetune/SOURCES.md` is written by the build with the same table.

The face-swap set is **same-source only**. The second face-swap set,
`skyhawk000/semisynth-faceswap`, is a byte-for-byte duplicate of `bitmind/face-swap`, and
`kenjon/deep-fake-face-swap` declares no licence. Until Celeb-DF access or another licence-checked
face-swap set from a different source is added, every face-swap result is reported as
**same-source only**.

**Real images are FairFace; train, validation and test reals are disjoint images.** The build
splits by group (see below), so each FairFace image is in exactly one split.

### Leakage control: what it does and does not do

`finetune/prepare.py` hashes every image with a 64-bit difference hash (dHash). Images within
Hamming distance 6 are linked, and each linked cluster becomes one group. Groups are split, never
single images, so near-duplicates (the same face at different crops or resolutions) cannot cross
splits.

**This reduces identity leakage but does not remove it.** dHash catches near-duplicate pixels. It
does not recognise a person across different photos or face-swap outputs, so the same identity can
still appear in train and test. Replacing dHash with a face-embedding clustering would catch more,
and needs a face-recognition model that is not in the environment yet. The face-swap numbers
should be read with that in mind.

On the current build, dHash linked only about 1,250 of ~15,000 images into groups, so most images
are their own group. The build prints the group count so this stays visible.

### Build the dataset

```bash
cd backend
python -m finetune.prepare --out ../.data/finetune --reals 5000 --seed 13
```

This downloads the two sources from the Hub, writes `images/real`, `images/fake`, `manifest.csv`,
`dataset.json` and `SOURCES.md`, and checks the splits. It needs about 1 GB of disk and several
minutes.

The same command builds the evaluation folders (`--eval-only` rebuilds only those from an existing manifest). They are built from the **test split only**: `eval/faceswap_same_source/` (513 real,
925 fake) and `eval/ai_faces_regression/` (90 FairFace test reals, 90 LucaCorvitto fakes).

## Commands

Run from `backend/`. On Windows, use `..\.venv\Scripts\python.exe` if you use the shared venv.

### 1. Install

```bash
pip install -r ../requirements.txt
pip install peft==0.21.2 mlflow-skinny==3.17.0   # the finetune extra, as pinned in pyproject.toml
```

### 2. Smoke test (CPU, a few minutes on a small manifest)

```bash
python scripts/finetune_lora.py --manifest ../.data/finetune/manifest.csv --out runs/smoke \
    --epochs 1 --batch-size 4 --workers 0 --max-train-steps 3
```

### 3. Full training, CPU

Slow: a ViT-S at 384 px with batch 16 takes several seconds per step on a 12-thread CPU.

```bash
python scripts/finetune_lora.py --manifest ../.data/finetune/manifest.csv --out runs/lora-1 \
    --epochs 3 --batch-size 16 --lr 1e-4 --rank 8 --alpha 16 --device cpu --mlflow
```

### 4. Full training, Google Colab (free GPU)

In Colab: **Runtime → Change runtime type → T4 GPU**. Then, in cells:

```bash
!git clone -b upgrade/v2 https://github.com/MuthuVarshith/FusionTrace.git
%cd FusionTrace
!pip install -q -r requirements.txt peft==0.21.2 mlflow-skinny==3.17.0
%cd backend
!python scripts/download_models.py                          # the default commfor weights from the Hub
!python -m finetune.prepare --out data --reals 5000 --seed 13
!python scripts/finetune_lora.py --manifest data/manifest.csv --out runs/lora-1 --epochs 3 --batch-size 16 --device cuda --mlflow
```

Copy `runs/lora-1` to Drive before the session ends (`!cp -r runs/lora-1 /content/drive/MyDrive/`).
Colab's MLflow store is local to the session; to keep runs, set `MLFLOW_TRACKING_URI` to a folder in
Drive before training.

### 5. Evaluate base vs fine-tuned

```bash
python scripts/evaluate_finetune.py --adapter runs/lora-1 \
    --set faceswap_same_source=data/eval/faceswap_same_source/real:data/eval/faceswap_same_source/fake \
    --set ai_faces_regression=data/eval/ai_faces_regression/real:data/eval/ai_faces_regression/fake \
    --out runs/lora-1/eval.json
```

The output is a Markdown table with AUC, fake recall and real recall for the base model (threshold
0.5) and the fine-tune (threshold from `threshold.json`). The face-swap rows are marked
`same-source only`.

### 6. Use the adapter in the app (off by default)

In `backend/config.yaml`, under `detectors.image.options`:

```yaml
lora:
  enabled: true
  source: mlflow            # or: path, with path: runs/lora-1
  model_name: fusiontrace-image-lora
  alias: candidate
```

The adapter's LoRA weights are merged into the ViT at load time, so inference and the heatmap
work as they do for the base model. The threshold comes from the adapter's run unless
`threshold` is set explicitly. `/health` and each scan's details show the adapter's version and
data version. Only the `candidate` alias is used here; promote it deliberately (step 7).

### 7. Promotion (manual, not part of this pipeline)

Only after reviewing the evaluation:

```bash
python -c "from mlops import model_registry as r; print(r.promote('fusiontrace-image-lora', 'candidate', 'production'))"
```

Then set `alias: production` under `lora` in `config.yaml`.

## Outputs

```
runs/lora-1/
  adapter/          PEFT adapter (LoRA weights + trained head)
  threshold.json    threshold, validation metrics, best epoch, data_version, args, history
  manifest.csv      copy of the manifest used
  eval.json         (after evaluate_finetune.py) per-set metrics for base and fine-tune
```

## Known limits

- **Blocked: a second face-swap test set.** Only the same-source set is available. Results on it say
  nothing about other face-swap methods.
- **Identity leakage is reduced, not removed** (see above). Validation and test face-swap numbers
  are optimistic to that extent.
- The LoRA target is the attention `qkv` and `proj` layers (221,569 trainable parameters, about 1%
  of the model, including the head). Augmentation is a random crop and horizontal flip only.
- The base model's diffusion-face AUC was 1.000, but its training data includes that generator, so
  the regression set is not an unseen-generator test. Its face-swap AUC before fine-tuning was
  0.459.
- The smoke run used a manifest of local test images, not the built dataset. The full dataset has
  not been trained on in this environment.
