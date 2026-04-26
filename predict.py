import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.similis_baseline.inference import predict_directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    predict_directory(args.image_dir, args.checkpoint, args.output_csv, device=args.device)


if __name__ == "__main__":
    main()
