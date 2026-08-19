import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50ImageEncoder(nn.Module):
    """ResNet50 image encoder with optional pretrained weights."""

    def __init__(self, finetune: bool = False, device: torch.device='cuda'):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.finetune = finetune
        self.output_dim = 2048  # ResNet50 final feature dimension
        self.device = device

        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        if not finetune:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings.

        Args:
            images: Tensor of shape (batch_size, C, H, W)

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Normalize images for ResNet (ImageNet stats)
        images = images.to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = (images - mean) / std

        features = self.features(images)  # (B, 2048, H, W)
        features = features.flatten(1)  # (B, 2048)

        return features