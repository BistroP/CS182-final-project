
"""Inspect Flickr8k splits and captions."""

from datasets import load_dataset

from spectral.multimodal_data import Flickr8kPairs, get_captions
import hashlib


def image_hash(image):
    # Hash the decoded RGB pixels
    rgb = image.convert("RGB")
    return hashlib.sha256(
        rgb.size[0].to_bytes(4, "big")
        + rgb.size[1].to_bytes(4, "big")
        + rgb.tobytes()
    ).hexdigest()

def main():
    dataset = load_dataset("intro/flickr8k")

    expected_sizes = {
        "train": 6000,
        "dev": 1000,
        "test": 1000,
    }

    print("Dataset splits:")
    for split, expected in expected_sizes.items():
        actual = len(dataset[split])
        print(f"  {split}: {actual} images")
        assert actual == expected


    image_hashes = {}

    for split in ["train", "dev", "test"]:
        hashes = {
            image_hash(example["image"])
            for example in dataset[split]
        }

        image_hashes[split] = hashes
        print(f"{split}: {len(hashes)} unique images")

    assert image_hashes["train"].isdisjoint(image_hashes["dev"])
    assert image_hashes["train"].isdisjoint(image_hashes["test"])
    assert image_hashes["dev"].isdisjoint(image_hashes["test"])

    print("No identical images across splits.")



    # Check that every image has five captions
    for split in expected_sizes:
        for example in dataset[split]:
            assert len(get_captions(example)) == 5

    print("All examples have five non-empty captions.")

    # Inspect a single image-caption pair
    pairs = Flickr8kPairs(dataset["train"])
    example = pairs[0]

    print("\nSample:")
    print("Image ID:", example["image_id"])
    print("Image size:", example["image"].size)
    print("Caption:", example["caption"])


if __name__ == "__main__":
    main()
