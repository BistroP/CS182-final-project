
"""Extract frozen ResNet-18 and MiniLM features from Flickr8k."""

import argparse
import hashlib
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torchvision.models import resnet18, ResNet18_Weights


TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def image_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(
        rgb.width.to_bytes(4, "big")
        + rgb.height.to_bytes(4, "big")
        + rgb.tobytes()
    ).hexdigest()


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(args):
    device = choose_device(args.device)
    print(f"Using device: {device}")

    # 1. Load dataset split
    dataset = load_dataset("intro/flickr8k", split=args.split)

    # 2. Load frozen encoders
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    image_encoder = resnet18(weights=weights)
    image_encoder.fc = torch.nn.Identity()
    image_encoder = image_encoder.to(device).eval()

    text_encoder = SentenceTransformer(
        TEXT_MODEL, device=device
    )
    text_encoder.eval()

    for p in image_encoder.parameters():
        p.requires_grad_(False)

    for p in text_encoder.parameters():
        p.requires_grad_(False)

    # 3. Collect unique images and captions
    rows = []
    seen_hashes = set()

    for idx, example in enumerate(dataset):
        image = example["image"].convert("RGB")
        h = image_hash(image)

        if h in seen_hashes:
            continue

        seen_hashes.add(h)

        captions = [
            example[f"caption_{i}"]
            for i in range(5)
        ]

        rows.append((idx, image, captions))

        if args.limit is not None and len(rows) >= args.limit:
            break

    print(f"Unique images selected: {len(rows)}")

    # 4. Extract image features in batches
    image_features = []

    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]

            images = torch.stack([
                preprocess(image) for _, image, _ in batch
            ]).to(device)

            features = image_encoder(images)
            image_features.append(features.cpu())

    image_features = torch.cat(image_features, dim=0)

    # 5. Extract all five captions per image
    captions = [
        caption
        for _, _, group in rows
        for caption in group
    ]

    text_features = text_encoder.encode(
        captions,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )

    text_features = text_features.cpu().reshape(
        len(rows), 5, -1
    )

    # 6. Validate output dimensions
    assert image_features.shape == (len(rows), 512)
    assert text_features.shape == (len(rows), 5, 384)
    assert torch.isfinite(image_features).all()
    assert torch.isfinite(text_features).all()

    # 7. Save features and their original row indices
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "image_features": image_features,
            "text_features": text_features,
            "row_indices": torch.tensor(
                [idx for idx, _, _ in rows]
            ),
            "split": args.split,
            "image_encoder": "resnet18-imagenet1k-v1",
            "text_encoder": TEXT_MODEL,
        },
        output,
    )

    print("\nExtraction successful!")
    print("Image features:", tuple(image_features.shape))
    print("Text features:", tuple(text_features.shape))
    print("Saved to:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default="data/processed/flickr8k_train_100.pt",
    )

    main(parser.parse_args())
