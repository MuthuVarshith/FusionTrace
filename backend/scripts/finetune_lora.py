"""LoRA fine-tune the default image detector (Community Forensics ViT-S/16-384).

    python scripts/finetune_lora.py --manifest data/manifest.csv --out runs/lora-1 --epochs 3

The manifest decides the splits (see finetune/data.py). Training uses the ``train``
rows; the best epoch is picked by ``val`` AUC; the decision threshold is picked on
``val`` only. ``test`` rows are never read here. Results go to ``--out``:

    adapter/            PEFT LoRA adapter (incl. the trained head)
    threshold.json      threshold, validation metrics, data version, args
    manifest.csv        copy of the manifest used

With ``--mlflow`` the run, params, metrics and data version are logged, and the
adapter is registered as model ``fusiontrace-image-lora`` with alias ``candidate``.
It is never given the ``production`` alias here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from finetune import data, model  # noqa: E402
from mlops.metrics import roc_auc  # noqa: E402

REGISTERED_NAME = "fusiontrace-image-lora"


def pick_threshold(risks: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    """Threshold on validation data that maximises balanced accuracy (ties: lower threshold)."""
    best_t, best_bal = 0.5, -1.0
    for t in np.unique(np.concatenate([risks, [0.0, 0.5, 1.0]])):
        pred = risks >= t
        fake_recall = float(pred[labels == 1].mean()) if (labels == 1).any() else 0.0
        real_recall = float((~pred[labels == 0]).mean()) if (labels == 0).any() else 0.0
        balanced = (fake_recall + real_recall) / 2
        if balanced > best_bal + 1e-12:
            best_t, best_bal = float(t), balanced
    pred = risks >= best_t
    return best_t, {
        "balanced_accuracy": best_bal,
        "fake_recall": float(pred[labels == 1].mean()) if (labels == 1).any() else 0.0,
        "real_recall": float((~pred[labels == 0]).mean()) if (labels == 0).any() else 0.0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None, help="base safetensors (default: the registry's commfor weights)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train-steps", type=int, default=0, help="stop after N steps (smoke runs only)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mlflow", action="store_true", help="log to MLflow and set the candidate alias")
    return parser.parse_args(argv)


def evaluate(net: torch.nn.Module, rows: list[data.Row], device: str, batch_size: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(data.ManifestDataset(rows, data.eval_transform()), batch_size=batch_size, num_workers=workers)
    return model.score(net, loader, device)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    args.out.mkdir(parents=True, exist_ok=True)

    rows = data.read_manifest(args.manifest)
    data.check_disjoint(rows)
    data.check_splits_present(rows)
    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    data_version = data.content_hash(rows)
    print(f"device={device} train={len(train_rows)} val={len(val_rows)} data_version={data_version[:12]}")

    net = model.build_lora(args.rank, args.alpha, args.dropout, args.weights).to(device)
    stats = model.trainable_summary(net)
    print(f"trainable {stats['trainable_params']:,} of {stats['total_params']:,} parameters")

    train_loader = DataLoader(
        data.ManifestDataset(train_rows, data.train_transform()),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False,
    )
    params = [p for p in net.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    best = {"auc": -1.0, "state": None, "epoch": 0}
    step = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        losses = []
        for pixels, targets in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(net(pixels.to(device)).reshape(-1), targets.to(device))
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
            step += 1
            if args.max_train_steps and step >= args.max_train_steps:
                break
        val_risk, val_labels = evaluate(net, val_rows, device, args.batch_size, args.workers)
        auc = roc_auc(val_labels.tolist(), val_risk.tolist())
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"), "val_auc": auc or float("nan")}
        history.append(entry)
        print(json.dumps(entry))
        if auc is not None and auc > best["auc"]:
            best = {"auc": auc, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if "lora_" in k or model.HEAD in k}}
        if args.max_train_steps and step >= args.max_train_steps:
            break

    if best["state"] is not None:
        net.load_state_dict(best["state"], strict=False)
    val_risk, val_labels = evaluate(net, val_rows, device, args.batch_size, args.workers)
    threshold, val_at_threshold = pick_threshold(val_risk, val_labels)

    adapter_dir = args.out / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    net.save_pretrained(str(adapter_dir))
    shutil.copy(args.manifest, args.out / "manifest.csv")
    summary = {
        "base": "OwensLab/commfor-model-384 @ 6076002bf0d9dd37537f965ee2f06f826c333b61",
        "threshold": threshold,
        "threshold_chosen_on": "val",
        "val_at_threshold": val_at_threshold,
        "best_epoch": best["epoch"],
        "best_val_auc": best["auc"],
        "history": history,
        "data_version": data_version,
        "counts": {s: sum(1 for r in rows if r.split == s) for s in data.SPLITS},
        "args": {k: str(v) for k, v in vars(args).items()},
        "trainable": stats,
        "seconds": round(time.time() - started, 1),
        "device": device,
        "smoke_run": bool(args.max_train_steps),
    }
    (args.out / "threshold.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"threshold": threshold, "best_val_auc": best["auc"], **val_at_threshold}))

    if args.mlflow:
        log_to_mlflow(args, summary, adapter_dir)
    return 0


def log_to_mlflow(args: argparse.Namespace, summary: dict, adapter_dir: Path) -> None:
    """Log the run and register the adapter as a new version with alias ``candidate``.

    Never sets ``production``: promotion is a separate, deliberate step.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    from mlflow.tracking import MlflowClient

    from mlops import model_registry

    mlflow.set_experiment(model_registry.EXPERIMENT)
    with mlflow.start_run(run_name=f"lora-{summary['data_version'][:8]}") as active:
        mlflow.log_params({
            "base": summary["base"], "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
            "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
            "best_epoch": summary["best_epoch"], "threshold": summary["threshold"],
            "data_version": summary["data_version"],
        })
        for entry in summary["history"]:
            mlflow.log_metrics({"train_loss": entry["train_loss"], "val_auc": entry["val_auc"]}, step=entry["epoch"])
        metrics = {f"val_{k}": v for k, v in summary["val_at_threshold"].items()}
        metrics["val_auc_best"] = summary["best_val_auc"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(args.out / "threshold.json"))
        mlflow.log_artifact(str(args.out / "manifest.csv"))
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
        run_id = active.info.run_id
        artifact_uri = active.info.artifact_uri

    client = MlflowClient()
    try:
        client.get_registered_model(REGISTERED_NAME)
    except Exception:  # not created yet; the exception type differs across stores
        client.create_registered_model(REGISTERED_NAME, description="LoRA adapters for the FusionTrace image detector (candidates).")
    version = client.create_model_version(
        REGISTERED_NAME, source=f"{artifact_uri}/adapter", run_id=run_id,
        tags={"data_version": summary["data_version"], "threshold": str(summary["threshold"]), "base": summary["base"]},
    )
    client.set_registered_model_alias(REGISTERED_NAME, "candidate", version.version)
    print(f"MLflow run {run_id}: {REGISTERED_NAME} v{version.version} has alias 'candidate' (no production alias set)")


if __name__ == "__main__":
    raise SystemExit(main())
