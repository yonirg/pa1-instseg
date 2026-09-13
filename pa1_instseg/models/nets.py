"""As três formas de recuperar resolução vistas em aula, sobre o MESMO encoder
(mesmas larguras ``base * 2**i``), para a ablação do Eixo 1:

* ``UNet``    — skip connections por concatenação (slides 25–29);
* ``SegNet``  — max-unpooling com os índices do pooling (slides 14 e 16);
* ``DeepLab`` — atrous mantendo output stride 8 + ASPP + bilinear (slides 39–42).

Todas recebem ``out_ch`` (2 para binário, 4 para fronteira+distância) e um
módulo de contexto global opcional no gargalo (Eixo 3). ``rf_layers()``
devolve a sequência (k, s, d) do caminho encoder → gargalo para o cálculo do
campo receptivo teórico.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ASPP, ConvBlock, make_context


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=16, depth=4, context="none"):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.enc.append(ConvBlock(c, ch)); c = ch
        self.bottleneck = ConvBlock(c, chs[-1])
        self.context = make_context(context, chs[-1])
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        c = chs[-1]
        for ch in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.dec.append(ConvBlock(2 * ch, ch)); c = ch
        self.head = nn.Conv2d(c, out_ch, 1)
        self.depth = depth

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x); skips.append(x); x = F.max_pool2d(x, 2)
        x = self.context(self.bottleneck(x))
        for up, blk, s in zip(self.up, self.dec, reversed(skips)):
            x = blk(torch.cat([up(x), s], 1))
        return self.head(x)

    def rf_layers(self, **_):
        rf = []
        for blk in self.enc:
            rf += blk.rf + [(2, 2, 1)]
        return rf + self.bottleneck.rf


class SegNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=16, depth=4, context="none"):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.enc.append(ConvBlock(c, ch)); c = ch
        self.bottleneck = ConvBlock(c, chs[-1])
        self.context = make_context(context, chs[-1])
        # o unpool do último estágio precisa de chs[depth-1] canais (os do índice)
        self.reduce = nn.Conv2d(chs[-1], chs[-2], 1)
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            cout = chs[i - 1] if i > 0 else chs[0]
            self.dec.append(ConvBlock(chs[i], cout))
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.depth = depth

    def forward(self, x):
        idxs, sizes = [], []
        for blk in self.enc:
            x = blk(x); sizes.append(x.shape[-2:])
            x, idx = F.max_pool2d(x, 2, return_indices=True); idxs.append(idx)
        x = self.reduce(self.context(self.bottleneck(x)))
        for blk, idx, sz in zip(self.dec, reversed(idxs), reversed(sizes)):
            x = F.max_unpool2d(x, idx, 2, output_size=sz)  # pool indices (slides 14/16)
            x = blk(x)
        return self.head(x)

    def rf_layers(self, **_):
        rf = []
        for blk in self.enc:
            rf += blk.rf + [(2, 2, 1)]
        return rf + self.bottleneck.rf


class DeepLab(nn.Module):
    """Encoder igual, mas a partir de ``output_stride`` os blocos usam dilatação
    (2, 4, …) no lugar de pooling. ASPP no fim e upsample bilinear ×OS."""

    def __init__(self, in_ch=1, out_ch=2, base=16, depth=4, output_stride=8,
                 aspp_rates=(2, 4, 8), context="none", atrous=True):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        n_pool = int(math.log2(output_stride))
        assert n_pool <= depth, "output_stride maior que o encoder permite"
        self.enc = nn.ModuleList()
        self.pool_flags = []
        c, d = in_ch, 1
        for i, ch in enumerate(chs[:-1]):
            if i < n_pool:
                self.enc.append(ConvBlock(c, ch, dilation=1)); self.pool_flags.append(True)
            else:
                d = d * 2  # no lugar do pooling: dobra a dilatação (mesma RF, mesma resolução)
                self.enc.append(ConvBlock(c, ch, dilation=d if atrous else 1)); self.pool_flags.append(False)
            c = ch
        d_b = d * 2 if (atrous and depth > n_pool) else 1
        self.bottleneck = ConvBlock(c, chs[-1], dilation=d_b)
        self.context = make_context(context, chs[-1])
        self.aspp = ASPP(chs[-1], chs[-1] // 2, rates=aspp_rates if atrous else (1, 1, 1))
        self.head = nn.Conv2d(chs[-1] // 2, out_ch, 1)
        self.output_stride, self.atrous = output_stride, atrous

    def forward(self, x):
        size = x.shape[-2:]
        for blk, pool in zip(self.enc, self.pool_flags):
            x = blk(x)
            if pool:
                x = F.max_pool2d(x, 2)
        x = self.aspp(self.context(self.bottleneck(x)))
        x = self.head(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def rf_layers(self, atrous: bool | None = None, **_):
        """``atrous=False`` responde 'e se eu tirasse a dilatação mantendo a
        mesma resolução de saída?' (Parte 5)."""
        use = self.atrous if atrous is None else atrous
        rf = []
        for blk, pool in zip(self.enc, self.pool_flags):
            rf += [(k, s, d if use else 1) for k, s, d in blk.rf]
            if pool:
                rf.append((2, 2, 1))
        rf += [(k, s, d if use else 1) for k, s, d in self.bottleneck.rf]
        rf += [(k, s, d if use else 1) for k, s, d in self.aspp.rf]
        return rf


class ResNetUNet(nn.Module):
    """U-Net com encoder ResNet34 pré-treinado em ImageNet (permitido pelo PA).
    Requer torchvision; ``pretrained=False`` para testes sem download."""

    def __init__(self, in_ch=3, out_ch=2, pretrained=True, context="none"):
        super().__init__()
        import torchvision
        w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet34(weights=w)
        if in_ch != 3:
            r.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # /2, 64
        self.pool = r.maxpool                                       # /4
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64,128,256,512
        self.context = make_context(context, 512)
        self.up4, self.d4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.d3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.d2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.d1 = nn.ConvTranspose2d(64, 64, 2, 2), ConvBlock(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d0 = ConvBlock(32, 32)
        self.head = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        s0 = self.stem(x)               # /2
        s1 = self.l1(self.pool(s0))     # /4
        s2 = self.l2(s1)                # /8
        s3 = self.l3(s2)                # /16
        s4 = self.context(self.l4(s3))  # /32
        x = self.d4(torch.cat([self.up4(s4), s3], 1))
        x = self.d3(torch.cat([self.up3(x), s2], 1))
        x = self.d2(torch.cat([self.up2(x), s1], 1))
        x = self.d1(torch.cat([self.up1(x), s0], 1))
        x = self.d0(self.up0(x))
        return self.head(x)

    def rf_layers(self, **_):
        # ResNet34: stem 7x7/2, pool 3x3/2, blocos [3,4,6,3] de 2 conv3x3, strides 2 nos layers 2-4
        rf = [(7, 2, 1), (3, 2, 1)]
        for n, s in zip([3, 4, 6, 3], [1, 2, 2, 2]):
            for i in range(n):
                rf += [(3, s if i == 0 else 1, 1), (3, 1, 1)]
        return rf


def build_model(arch: str, in_ch: int, out_ch: int, base: int = 16, depth: int = 4,
                context: str = "none", pretrained: bool = True, **kw) -> nn.Module:
    if arch == "unet":
        return UNet(in_ch, out_ch, base, depth, context)
    if arch == "segnet":
        return SegNet(in_ch, out_ch, base, depth, context)
    if arch == "deeplab":
        return DeepLab(in_ch, out_ch, base, depth, context=context, **kw)
    if arch == "resnet34_unet":
        return ResNetUNet(in_ch, out_ch, pretrained=pretrained, context=context)
    raise ValueError(f"arquitetura desconhecida: {arch}")
