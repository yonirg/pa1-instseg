"""Blocos compartilhados. Cada bloco expõe ``rf`` = lista de (kernel, stride, dilation)
na ordem em que os pixels de entrada são consumidos, para o cálculo do campo
receptivo teórico (slides 35–38) em ``receptive_field.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """(conv3x3 → BN → ReLU) × 2, com dilatação opcional (atrous)."""

    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        p = dilation
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=p, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=p, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
        self.rf = [(3, 1, dilation), (3, 1, dilation)]

    def forward(self, x):
        return self.net(x)


class ImagePooling(nn.Module):
    """ParseNet (slides 51–53): média global → 1x1 → broadcast → concat → 1x1."""

    def __init__(self, ch: int):
        super().__init__()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch, 1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(2 * ch, ch, 1, bias=False), nn.BatchNorm2d(ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        g = F.interpolate(self.pool(x), size=x.shape[-2:], mode="nearest")
        return self.fuse(torch.cat([x, g], 1))


class PSPModule(nn.Module):
    """PSPNet pyramid pooling (slides 54–55) com bins (1,2,3,6)."""

    def __init__(self, ch: int, bins=(1, 2, 3, 6)):
        super().__init__()
        red = max(ch // len(bins), 8)
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(ch, red, 1, bias=False),
                          nn.BatchNorm2d(red), nn.ReLU(inplace=True)) for b in bins])
        self.fuse = nn.Sequential(nn.Conv2d(ch + red * len(bins), ch, 1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

    def forward(self, x):
        outs = [x] + [F.interpolate(s(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
                      for s in self.stages]
        return self.fuse(torch.cat(outs, 1))


class ASPP(nn.Module):
    """DeepLab v3 ASPP (slides 39–42): 1x1 + 3 atrous 3x3 + image pooling."""

    def __init__(self, cin: int, cout: int, rates=(2, 4, 8)):
        super().__init__()
        def br(k, d):
            return nn.Sequential(nn.Conv2d(cin, cout, k, padding=(k // 2) * d, dilation=d, bias=False),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.branches = nn.ModuleList([br(1, 1)] + [br(3, r) for r in rates])
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False),
                                        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(cout * (len(rates) + 2), cout, 1, bias=False),
                                     nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.rates = rates
        # RF da branch mais dilatada (o pior caso é o que interessa)
        self.rf = [(3, 1, max(rates))]

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        outs.append(F.interpolate(self.image_pool(x), size=x.shape[-2:], mode="nearest"))
        return self.project(torch.cat(outs, 1))


def make_context(kind: str, ch: int) -> nn.Module:
    if kind in (None, "none"):
        return nn.Identity()
    if kind == "image_pooling":
        return ImagePooling(ch)
    if kind == "psp":
        return PSPModule(ch)
    raise ValueError(f"contexto desconhecido: {kind}")
