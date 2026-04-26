from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.similis_baseline import build_features, build_split_frames, build_label_vocabs, SimilisDataset, collate_fn, make_transforms
from src.similis_baseline.inference import load_model_from_checkpoint
from src.similis_baseline.template import build_auto_description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/interim/manifest_raw.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train_inner", "val_inner", "test_open"], required=True)
    parser.add_argument("--output-preds", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    df = build_features(args.manifest)
    train_df, val_df, test_df = build_split_frames(df, seed=42)
    split_map = {"train_inner": train_df, "val_inner": val_df, "test_open": test_df}
    split_df = split_map[args.split].copy().reset_index(drop=True)

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device=device)
    fields = ckpt["fields"]
    _, field_to_idx, idx_to_field = build_label_vocabs(df, fields)
    _, val_tf = make_transforms(ckpt.get("config", {}).get("image_size", 224), train_aug=False)
    ds = SimilisDataset(split_df, val_tf, field_to_idx, fields)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)
    thresholds = ckpt.get("template_config", {}).get("thresholds", {})

    rows = []
    all_true = {f: [] for f in fields}
    all_pred = {f: [] for f in fields}
    all_mask = {f: [] for f in fields}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            probs = {field: torch.softmax(outputs[field], dim=1).cpu() for field in fields}
            for i in range(images.shape[0]):
                row = {
                    "image_file": batch["metadata"]["image_file"][i],
                    "group_key": batch["metadata"]["group_key"][i],
                }
                pred_labels = {}
                conf = {}
                for field in fields:
                    pred_idx = int(probs[field][i].argmax().item())
                    pred_label = idx_to_field[field][pred_idx]
                    gt_idx = int(batch["targets"][field][i].item())
                    gt_label = idx_to_field[field][gt_idx]
                    mask_value = float(batch["target_mask"][field][i].item())
                    confidence = float(probs[field][i][pred_idx].item())

                    row[f"gt_{field}"] = gt_label
                    row[f"pred_{field}"] = pred_label
                    row[f"confidence_{field}"] = confidence
                    row[f"mask_{field}"] = mask_value

                    pred_labels[field] = pred_label
                    conf[field] = confidence
                    all_true[field].append(gt_label)
                    all_pred[field].append(pred_label)
                    all_mask[field].append(mask_value)

                row["auto_description"] = build_auto_description(pred_labels, conf=conf, thresholds=thresholds)
                rows.append(row)

    pred_df = pd.DataFrame(rows)
    Path(args.output_preds).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_preds, index=False)

    metrics = {}
    macro_values = []
    for field in fields:
        mask = [m > 0 for m in all_mask[field]]
        y_true = [y for y, keep in zip(all_true[field], mask) if keep]
        y_pred = [y for y, keep in zip(all_pred[field], mask) if keep]
        acc = accuracy_score(y_true, y_pred) if y_true else float("nan")
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else float("nan")
        metrics[field] = {"accuracy": acc, "macro_f1": macro_f1}
        macro_values.append(macro_f1)
    metrics["mean_macro_f1"] = float(sum(macro_values) / len(macro_values))

    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
