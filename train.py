import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.similis_baseline import (
    SimilisDataset,
    MultiTaskArtifactModel,
    build_features,
    build_label_vocabs,
    build_split_frames,
    collate_fn,
    get_default_config,
    make_transforms,
)
from src.similis_baseline.training import run_epoch, save_checkpoint, append_log


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default="data/interim/manifest_raw.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts/checkpoints/baseline_run")
    parser.add_argument("--tiny-overfit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_default_config()
    if args.config:
        config.update(json.loads(Path(args.config).read_text()))

    set_seed(config["seed"])
    device = torch.device(config.get("device", "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_features(args.manifest)
    train_inner_df, val_inner_df, test_open_df = build_split_frames(df, seed=config["seed"])
    fields = config["fields"]
    _, field_to_idx, idx_to_field = build_label_vocabs(df, fields)
    train_tf, val_tf = make_transforms(config["image_size"], train_aug=config["train_aug"])

    train_ds = SimilisDataset(train_inner_df, train_tf, field_to_idx, fields)
    val_ds = SimilisDataset(val_inner_df, val_tf, field_to_idx, fields)

    if args.tiny_overfit:
        subset_size = min(config["tiny_subset_size"], len(train_ds))
        subset_indices = list(range(subset_size))
        train_ds = Subset(train_ds, subset_indices)
        val_ds = Subset(val_ds, subset_indices[: min(len(val_ds), subset_size // 2)])
        epochs = config["tiny_epochs"]
    else:
        epochs = config["epochs"]

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )

    field_dims = {field: len(field_to_idx[field]) for field in fields}
    model = MultiTaskArtifactModel(
        field_dims=field_dims,
        backbone_name=config["backbone"],
        pretrained=config["pretrained"],
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    class_weights = {}
    if config.get("use_class_weights", False):
        for field in fields:
            counts = train_inner_df[field].value_counts()
            weights = []
            for idx in range(len(field_to_idx[field])):
                label = idx_to_field[field][idx]
                count = counts.get(label, 1)
                weights.append(1.0 / count)
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            weight_tensor = weight_tensor / weight_tensor.mean()
            class_weights[field] = weight_tensor

    best_metric = -1.0
    log_path = output_dir / "train_log.csv"
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, fields, class_weights, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, fields, class_weights, train=False)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mean_macro_f1": val_metrics["mean_macro_f1"],
        }
        for field in fields:
            row[f"train_acc_{field}"] = train_metrics["per_field"][field]["accuracy"]
            row[f"train_f1_{field}"] = train_metrics["per_field"][field]["macro_f1"]
            row[f"val_acc_{field}"] = val_metrics["per_field"][field]["accuracy"]
            row[f"val_f1_{field}"] = val_metrics["per_field"][field]["macro_f1"]
        append_log(log_path, row)

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            field_to_idx,
            idx_to_field,
            config["template_config"],
            fields,
            config,
        )
        if val_metrics["mean_macro_f1"] > best_metric:
            best_metric = val_metrics["mean_macro_f1"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                field_to_idx,
                idx_to_field,
                config["template_config"],
                fields,
                config,
            )

    summary = {
        "output_dir": str(output_dir),
        "best_metric": best_metric,
        "fields": fields,
        "config": config,
        "num_parameters": model.num_parameters,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
