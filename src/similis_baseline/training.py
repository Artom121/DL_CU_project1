import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def masked_cross_entropy(logits, targets, mask, class_weights=None):
    per_sample = F.cross_entropy(logits, targets, reduction="none", weight=class_weights)
    weighted = per_sample * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def compute_batch_metrics(logits, targets, mask):
    preds = logits.argmax(dim=1)
    valid = mask > 0
    if valid.sum().item() == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    y_true = targets[valid].detach().cpu().numpy()
    y_pred = preds[valid].detach().cpu().numpy()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_epoch(model, loader, optimizer, device, fields, class_weights=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    loss_sums = {field: 0.0 for field in fields}
    all_true = {field: [] for field in fields}
    all_pred = {field: [] for field in fields}
    all_mask = {field: [] for field in fields}

    for batch in loader:
        images = batch["image"].to(device)
        targets = {field: batch["targets"][field].to(device) for field in fields}
        target_mask = {field: batch["target_mask"][field].to(device) for field in fields}

        with torch.set_grad_enabled(train):
            outputs = model(images)
            field_losses = {}
            for field in fields:
                cw = class_weights.get(field) if class_weights else None
                field_losses[field] = masked_cross_entropy(outputs[field], targets[field], target_mask[field], cw)
            loss = sum(field_losses.values()) / len(fields)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        for field in fields:
            loss_sums[field] += field_losses[field].item()
            all_true[field].append(targets[field].detach().cpu())
            all_pred[field].append(outputs[field].argmax(dim=1).detach().cpu())
            all_mask[field].append(target_mask[field].detach().cpu())

    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "field_losses": {field: loss_sums[field] / max(len(loader), 1) for field in fields},
        "per_field": {},
    }

    for field in fields:
        y_true = torch.cat(all_true[field])
        y_pred = torch.cat(all_pred[field])
        mask = torch.cat(all_mask[field]) > 0
        if mask.sum().item() == 0:
            acc = float("nan")
            macro_f1 = float("nan")
        else:
            true_np = y_true[mask].numpy()
            pred_np = y_pred[mask].numpy()
            acc = accuracy_score(true_np, pred_np)
            macro_f1 = f1_score(true_np, pred_np, average="macro", zero_division=0)
        metrics["per_field"][field] = {"accuracy": acc, "macro_f1": macro_f1}

    macro_f1_values = [metrics["per_field"][field]["macro_f1"] for field in fields]
    metrics["mean_macro_f1"] = float(np.nanmean(macro_f1_values))
    return metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_metric,
    field_to_idx,
    idx_to_field,
    template_config,
    fields,
    config,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "field_to_idx": field_to_idx,
        "idx_to_field": idx_to_field,
        "template_config": template_config,
        "fields": fields,
        "config": config,
    }
    torch.save(payload, path)


def append_log(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
