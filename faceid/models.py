"""Embedding backbones and numerically stable ArcFace training head."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class FaceCNN(nn.Module):
    """Exact parameter names/shapes of the supplied 128-D legacy backbone."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_layer = nn.Linear(128, embedding_dim)

    def forward(self, x):
        return self.embedding_layer(torch.flatten(self.global_pool(self.features(x)), 1))

    def get_embedding(self, x):
        return F.normalize(self(x), dim=1)


class TinyCNN(nn.Module):
    """Small offline backbone for smoke tests and low-resource experiments."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(4, 32), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.embedding_layer = nn.Linear(32, embedding_dim)

    def forward(self, x):
        return self.embedding_layer(self.features(x).flatten(1))


class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        # Downloads occur only when the caller explicitly requests pretrained training.
        self.network = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        return self.network(x)


def build_backbone(architecture="resnet18", embedding_dim=128, pretrained=False):
    if not isinstance(embedding_dim, int) or embedding_dim < 2:
        raise ValueError("embedding_dim phải là số nguyên >= 2.")
    if architecture in {"facecnn", "legacy"}:
        return FaceCNN(embedding_dim)
    if architecture == "tinycnn":
        return TinyCNN(embedding_dim)
    if architecture == "resnet18":
        return ResNetEmbedding(embedding_dim, pretrained)
    raise ValueError(f"Backbone không được hỗ trợ: {architecture}")


class ArcFace(nn.Module):
    def __init__(self, embedding_dim, num_classes=3, margin=0.3, scale=30.0):
        super().__init__()
        if embedding_dim < 2 or num_classes < 2 or not 0 <= margin < math.pi / 2 or scale <= 0:
            raise ValueError("Tham số ArcFace không hợp lệ.")
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin, self.scale = float(margin), float(scale)
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th, self.mm = math.cos(math.pi - margin), math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        if not self.training or labels is None:
            return cosine * self.scale
        labels = labels.to(device=embeddings.device, dtype=torch.long)
        if labels.ndim != 1 or len(labels) != len(embeddings):
            raise ValueError("Nhãn phải có shape [batch_size].")
        if torch.any((labels < 0) | (labels >= self.weight.shape[0])):
            raise ValueError("Nhãn nằm ngoài số lớp ArcFace.")
        sine = (1 - cosine.square()).clamp_min(1e-7).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        target = F.one_hot(labels, self.weight.shape[0]).to(cosine.dtype)
        return (target * phi + (1 - target) * cosine) * self.scale
