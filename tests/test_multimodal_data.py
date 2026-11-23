
import pytest
from PIL import Image

from spectral.multimodal_data import Flickr8kPairs, get_captions


def make_example():
    return {
        "image": Image.new("RGB", (32, 32)),
        "file_name": "example.jpg",
        **{f"caption_{i}": f"Caption {i}" for i in range(5)},
    }


def test_get_captions():
    captions = get_captions(make_example())

    assert len(captions) == 5
    assert captions[0] == "Caption 0"


def test_caption_selection():
    dataset = Flickr8kPairs([make_example()], caption_index=2)

    assert len(dataset) == 1
    assert dataset[0]["caption"] == "Caption 2"

    dataset.set_caption_index(4)
    assert dataset[0]["caption"] == "Caption 4"


def test_invalid_caption_index():
    with pytest.raises(ValueError):
        Flickr8kPairs([make_example()], caption_index=5)
