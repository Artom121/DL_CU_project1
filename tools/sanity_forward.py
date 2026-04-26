from pathlib import Path
import sys

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.similis_baseline import (
    SimilisDataset,
    MultiTaskArtifactModel,
    build_features,
    build_label_vocabs,
    build_split_frames,
    collate_fn,
    make_transforms,
)
from src.similis_baseline.training import compute_batch_metrics, masked_cross_entropy


def main():
    fields = ["object_type", "integrity", "material_group", "part_zone"]
    df = build_features("data/interim/manifest_raw.csv")
    train_df, _, _ = build_split_frames(df, seed=42)
    _, field_to_idx, _ = build_label_vocabs(df, fields)
    train_tf, _ = make_transforms(224, train_aug=True)
    train_ds = SimilisDataset(train_df, train_tf, field_to_idx, fields)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = MultiTaskArtifactModel(
        {field: len(field_to_idx[field]) for field in fields},
        backbone_name="resnet18",
        pretrained=False,
    )
    outputs = model(batch["image"])

    print("image", tuple(batch["image"].shape), batch["image"].dtype)
    for field in fields:
        logits = outputs[field]
        targets = batch["targets"][field]
        mask = batch["target_mask"][field]
        loss = masked_cross_entropy(logits, targets, mask)
        metrics = compute_batch_metrics(logits, targets, mask)
        print(field, tuple(logits.shape), tuple(targets.shape), float(loss), metrics)
    print("params", model.num_parameters)


if __name__ == "__main__":
    main()
