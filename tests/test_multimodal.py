
import torch

from spectral.multimodal import MultimodalModel, contrastive_loss


def test_multimodal_forward_backward():
    torch.manual_seed(42)

    batch_size = 8
    model = MultimodalModel()

    # Fake image/text features for now
    images = torch.randn(batch_size, 512)
    texts = torch.randn(batch_size, 384)

    # Forward pass
    image_emb, text_emb = model(images, texts)

    assert image_emb.shape == (8, 256)
    assert text_emb.shape == (8, 256)

    # Compute loss
    loss = contrastive_loss(image_emb, text_emb)

    assert torch.isfinite(loss)

    # Backward pass
    loss.backward()

    # Check all trainable parameters received gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
