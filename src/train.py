"""Train one (model, preprocessing) experiment on the Niten Lama dataset.

    python -m src.train --model mobilenetv3small --preprocess D2

Two-phase transfer learning: train the classifier head with the backbone frozen
for ``--freeze-epochs``, then unfreeze and fine-tune at a lower LR. Early stopping
on validation macro-F1; the best checkpoint is used for the held-out test report.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CLASSES, FaceShapeDataset, load_split, make_split
from .evaluate import compute_metrics, save_confusion_matrix
from .models import build_model, set_backbone_trainable
from .preprocessing import build_transform, source_root_for

# The data split is a fixed property of the study, not a tunable. It stays at 42
# for every run: the near-duplicate audit was performed against this partition,
# so re-splitting would silently invalidate the deduplication. ``--seed`` controls
# training randomness only (weight init, shuffling, augmentation sampling).
SPLIT_SEED = 42


def environment_info(device):
    """Hardware/software provenance recorded with each run's metrics."""
    info = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_type": device.type,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update({
            "gpu": torch.cuda.get_device_name(device),
            "gpu_memory_total_gb": round(props.total_memory / 1024 ** 3, 2),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        })
    return info


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():   # Intel Arc
        return torch.device("xpu")
    return torch.device("cpu")


def make_loader(version, split, args, shuffle):
    src_root = source_root_for(version, args.data_root, args.preprocessed_root)
    items = load_split(args.split_csv, split, source_root=src_root)
    tfm = build_transform(version, split, img_size=args.img_size)
    ds = FaceShapeDataset(items, tfm)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True)


def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum, correct, total = 0.0, 0, 0
    preds, tgts = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        preds.append(pred.cpu().numpy())
        tgts.append(y.cpu().numpy())
    return loss_sum / total, correct / total, np.concatenate(preds), np.concatenate(tgts)


def optimizer_for(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preprocess", required=True,
                    choices=["D1", "D2", "D3", "D4", "D3n", "D4n"])
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--preprocessed-root", default="data/preprocessed")
    ap.add_argument("--split-csv", default="splits/split.csv")
    ap.add_argument("--output-dir", default="runs")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-full", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42,
                    help="training seed (weight init / shuffle / augmentation). "
                         "The data split always uses SPLIT_SEED and is unaffected.")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"[device] {device}")

    if not Path(args.split_csv).exists():
        print(f"[split] creating {args.split_csv} (split seed={SPLIT_SEED})")
        make_split(args.data_root, args.split_csv, seed=SPLIT_SEED)

    train_loader = make_loader(args.preprocess, "train", args, shuffle=True)
    val_loader = make_loader(args.preprocess, "val", args, shuffle=False)
    test_loader = make_loader(args.preprocess, "test", args, shuffle=False)

    model = build_model(args.model, num_classes=len(CLASSES)).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    exp = f"{args.model}_{args.preprocess}"
    out = Path(args.output_dir) / exp
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "epochs.csv"
    log_path.write_text("epoch,phase,train_loss,train_acc,val_loss,val_acc,val_macro_f1\n",
                        encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()

    best_f1, best_state, no_improve = -1.0, None, 0
    epochs_run = 0
    for epoch in range(args.epochs):
        epochs_run = epoch + 1
        if epoch < args.freeze_epochs:
            set_backbone_trainable(model, False)
            phase, opt = "head", optimizer_for(model, args.lr_head, args.weight_decay)
        else:
            set_backbone_trainable(model, True)
            phase, opt = "full", optimizer_for(model, args.lr_full, args.weight_decay)

        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, device, criterion, opt)
        va_loss, va_acc, va_pred, va_tgt = run_epoch(model, val_loader, device, criterion)
        vm = compute_metrics(va_tgt, va_pred)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{phase},{tr_loss:.4f},{tr_acc:.4f},"
                    f"{va_loss:.4f},{va_acc:.4f},{vm['macro_f1']:.4f}\n")
        print(f"[{exp}] e{epoch} {phase} val_acc={va_acc:.3f} val_f1={vm['macro_f1']:.3f}")

        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out / "best.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience and epoch >= args.freeze_epochs:
                print(f"[{exp}] early stop at epoch {epoch}")
                break

    train_seconds = time.perf_counter() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)
    _, te_acc, te_pred, te_tgt = run_epoch(model, test_loader, device, criterion)
    metrics = compute_metrics(te_tgt, te_pred)
    metrics.update({
        "model": args.model,
        "preprocess": args.preprocess,
        "test_accuracy": te_acc,
        "best_val_macro_f1": best_f1,
        "seed": args.seed,
        "split_seed": SPLIT_SEED,
        "epochs_run": epochs_run,
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.perf_counter() - t_start, 1),
        "environment": environment_info(device),
    })
    if device.type == "cuda":
        metrics["peak_gpu_memory_gb"] = round(
            torch.cuda.max_memory_allocated(device) / 1024 ** 3, 2)

    (out / "results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Paired test-set predictions, so McNemar can compare this run against any
    # other without reloading checkpoints (cf. journal_stats.py).
    np.savez(out / "test_predictions.npz", y_true=te_tgt, y_pred=te_pred)
    save_confusion_matrix(te_tgt, te_pred, CLASSES, out / "confusion_matrix.png")
    print(f"[{exp}] DONE test_acc={te_acc:.3f} test_macro_f1={metrics['macro_f1']:.3f} "
          f"({train_seconds / 60:.1f} min, {epochs_run} epochs)")


if __name__ == "__main__":
    main()
