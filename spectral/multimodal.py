
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalModel(nn.Module):
    def __init__(
        self,
        image_dim=512,
        text_dim=384,
        hidden_dim=512,
        embed_dim=256,
    ):
        super().__init__()

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, image_features, text_features):
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        return image_embeddings, text_embeddings


def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    logits = image_embeddings @ text_embeddings.T / temperature

    labels = torch.arange(
        logits.size(0),
        device=logits.device,
    )

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2
