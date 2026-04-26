from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
import torchvision.transforms as T

from .labels import OBJECT_TYPE_MAP, MATERIAL_MAP, INTEGRITY_MAP, PART_ZONE_RULES

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def extract_part_zone(text: str) -> str:
    text = (text or "").lower()
    for needle, label in PART_ZONE_RULES:
        if needle in text:
            return label
    return "unknown"


def build_features(manifest_path):
    df = pd.read_csv(manifest_path)
    df = df[df["image_exists"] == True].copy()
    df["description"] = df["description"].fillna("")
    df["artifact_id"] = df["artifact_id"].astype(str)
    df["group_key"] = df["group_key"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    df["object_type"] = df["name"].map(OBJECT_TYPE_MAP).fillna("other")
    df["material_group"] = df["material"].map(MATERIAL_MAP).fillna("other")
    df["integrity"] = df["fragm"].map(INTEGRITY_MAP).fillna("unknown")
    df["part_zone"] = df["description"].map(extract_part_zone)
    df["label_is_missing_part_zone"] = df["part_zone"].eq("unknown")
    df["label_is_missing_material"] = df["material"].isna()
    df["label_is_uncertain_object_type"] = df["name"].astype(str).str.contains(r"\?|/", regex=True)
    df["label_is_uncertain_part_zone"] = False
    return df.reset_index(drop=True)


def build_label_vocabs(df, fields):
    label_vocab = {field: sorted(df[field].dropna().unique().tolist()) for field in fields}
    field_to_idx = {field: {label: idx for idx, label in enumerate(values)} for field, values in label_vocab.items()}
    idx_to_field = {field: {idx: label for label, idx in mapping.items()} for field, mapping in field_to_idx.items()}
    return label_vocab, field_to_idx, idx_to_field


def build_split_frames(df, seed=42):
    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_outer.split(df, df["object_type"], df["group_key"]))
    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_open_df = df.iloc[test_idx].copy().reset_index(drop=True)

    sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(
        sgkf_inner.split(train_val_df, train_val_df["object_type"], train_val_df["group_key"])
    )
    train_inner_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_inner_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)
    return train_inner_df, val_inner_df, test_open_df


class ResizeLongestSidePad:
    def __init__(self, size, fill=(255, 255, 255)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        img = img.convert("RGB")
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - new_w) // 2
        top = (self.size - new_h) // 2
        canvas.paste(resized, (left, top))
        return canvas


def make_transforms(image_size=224, train_aug=True):
    train_steps = [ResizeLongestSidePad(image_size)]
    if train_aug:
        train_steps.extend(
            [
                T.RandomRotation(degrees=7, fill=255),
                T.ColorJitter(brightness=0.10, contrast=0.10),
            ]
        )
    train_steps.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_steps = [
        ResizeLongestSidePad(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(train_steps), T.Compose(val_steps)


class SimilisDataset(Dataset):
    def __init__(self, frame, transform, field_to_idx, fields):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        self.field_to_idx = field_to_idx
        self.fields = fields

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(img)
        targets = {}
        target_mask = {}
        for field in self.fields:
            value = row[field]
            targets[field] = self.field_to_idx[field][value]
            target_mask[field] = 0.0 if value == "unknown" else 1.0
        metadata = {
            "image_file": row["image_file"],
            "group_key": row["group_key"],
            "image_path": row["image_path"],
        }
        return {"image": image, "targets": targets, "target_mask": target_mask, "metadata": metadata}


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    target_cols = batch[0]["targets"].keys()
    targets = {col: torch.tensor([b["targets"][col] for b in batch], dtype=torch.long) for col in target_cols}
    target_mask = {
        col: torch.tensor([b["target_mask"][col] for b in batch], dtype=torch.float32) for col in target_cols
    }
    metadata = {k: [b["metadata"][k] for b in batch] for k in batch[0]["metadata"].keys()}
    return {"image": images, "targets": targets, "target_mask": target_mask, "metadata": metadata}
