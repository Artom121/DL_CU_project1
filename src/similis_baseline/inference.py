from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
import torch

from .data import ResizeLongestSidePad
from .model import MultiTaskArtifactModel
from .template import build_auto_description
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def make_eval_transform(image_size):
    return T.Compose(
        [
            ResizeLongestSidePad(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model_from_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    fields = ckpt["fields"]
    idx_to_field = ckpt["idx_to_field"]
    field_dims = {field: len(idx_to_field[field]) for field in fields}
    backbone = ckpt.get("config", {}).get("backbone", "resnet18")
    pretrained = False
    model = MultiTaskArtifactModel(field_dims=field_dims, backbone_name=backbone, pretrained=pretrained)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def predict_directory(image_dir, ckpt_path, output_csv, device="cpu"):
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)
    model, ckpt = load_model_from_checkpoint(ckpt_path, device=device)
    image_size = ckpt.get("config", {}).get("image_size", 224)
    transform = make_eval_transform(image_size)
    fields = ckpt["fields"]
    idx_to_field = ckpt["idx_to_field"]
    thresholds = ckpt.get("template_config", {}).get("thresholds", {})

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        image_paths.extend(sorted(image_dir.rglob(ext)))

    rows = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            outputs = model(tensor)
            probs = {field: torch.softmax(outputs[field], dim=1)[0].cpu() for field in fields}
            pred_idx = {field: int(prob.argmax().item()) for field, prob in probs.items()}
            pred_label = {field: idx_to_field[field][pred_idx[field]] for field in fields}
            conf = {field: float(probs[field][pred_idx[field]].item()) for field in fields}
            auto_description = build_auto_description(pred_label, conf=conf, thresholds=thresholds)
            row = {
                "image_file": str(path.relative_to(image_dir)),
                "auto_description": auto_description,
            }
            for field in fields:
                row[f"pred_{field}"] = pred_label[field]
                row[f"confidence_{field}"] = conf[field]
            rows.append(row)

    pred_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=False)
    return pred_df
