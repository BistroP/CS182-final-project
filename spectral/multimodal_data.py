
"""Dataset utilities for multimodal image-text experiments."""

from torch.utils.data import Dataset


CAPTION_COLUMNS = [f"caption_{i}" for i in range(5)]


def get_captions(example):
    """Return the five captions associated with an image."""
    captions = [example[column] for column in CAPTION_COLUMNS]

    if any(not isinstance(c, str) or not c.strip() for c in captions):
        raise ValueError("Missing or empty caption")

    return captions


class Flickr8kPairs(Dataset):
    """Return one image-caption pair per image.

    The caption index can be changed between epochs.
    """

    def __init__(self, hf_dataset, caption_index=0):
        self.dataset = hf_dataset
        self.set_caption_index(caption_index)

    def set_caption_index(self, index):
        if not 0 <= index < 5:
            raise ValueError("caption_index must be between 0 and 4")
        self.caption_index = index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[index]
        captions = get_captions(example)

        return {
            "image": example["image"].convert("RGB"),
            "caption": captions[self.caption_index],
            "image_id": index,
        }
