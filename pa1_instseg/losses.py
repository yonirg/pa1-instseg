"""Perdas da aula, escritas à mão.

* ``ce``              — entropia cruzada padrão (slide 73).
* ``balanced_ce``     — CE com peso α_c por classe (slide 74). ``alpha`` pode ser
                        um vetor fixo ou ``"auto"`` = frequência inversa
                        normalizada, estimada no *batch* (média móvel).
* ``focal``           — (1 − p_t)^γ · CE (slides 76–79). γ=0 recupera a CE.
* ``balanced_focal``  — α_t (1 − p_t)^γ · CE.
* ``l1`` / ``l2``     — regressão do mapa de distância (slide 80).

A ``MultiHeadLoss`` fatia os logits da rede: canais ``[:n_cls]`` são as classes
semânticas, o canal seguinte (se houver) é a distância (sigmoid ∈ [0,1]).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLoss(nn.Module):
    def __init__(self, n_classes: int, kind: str = "ce", gamma: float = 2.0,
                 alpha="auto", ema: float = 0.98, alpha_max: float | None = None):
        """``alpha_max`` limita o peso da classe minoritária: sem teto, a
        fronteira (~5% dos pixels) recebe α≈5× o do interior e redes pequenas
        colapsam para "todo objeto é fronteira" (IoU do interior = 0)."""
        super().__init__()
        assert kind in ("ce", "balanced_ce", "focal", "balanced_focal"), kind
        self.n_classes, self.kind, self.gamma, self.ema = n_classes, kind, gamma, ema
        self.alpha_max = alpha_max
        self.balanced = kind.startswith("balanced")
        if kind == "ce":
            self.gamma = 0.0
        if isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            self.auto = False
        else:
            self.register_buffer("alpha", torch.ones(n_classes))
            self.auto = True

    @torch.no_grad()
    def _update_alpha(self, target: torch.Tensor):
        freq = torch.bincount(target.flatten(), minlength=self.n_classes).float()
        freq = freq / freq.sum().clamp_min(1)
        inv = 1.0 / (freq + 1e-3)
        inv = inv / inv.sum() * self.n_classes           # média dos pesos = 1
        if self.alpha_max is not None:
            inv = inv.clamp(max=self.alpha_max)
        self.alpha.mul_(self.ema).add_((1 - self.ema) * inv)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)      # (B,H,W)
        ce = -logp_t
        if self.gamma > 0:
            p_t = logp_t.exp()
            ce = (1 - p_t).pow(self.gamma) * ce
        if self.balanced:
            if self.auto and self.training:
                self._update_alpha(target)
            ce = self.alpha[target] * ce
        return ce.mean()


class DistanceLoss(nn.Module):
    def __init__(self, kind: str = "l1"):
        super().__init__()
        assert kind in ("l1", "l2")
        self.kind = kind

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logit)
        if self.kind == "l1":
            return (pred - target).abs().mean()
        return ((pred - target) ** 2).mean()


class MultiHeadLoss(nn.Module):
    """head='binary': 2 logits, target sem2.   head='boundary': 3 logits + 1 dist."""

    def __init__(self, head: str, sem_kind: str = "ce", gamma: float = 2.0, alpha="auto",
                 dist_kind: str = "l1", dist_weight: float = 1.0, alpha_max: float | None = None):
        super().__init__()
        self.head = head
        self.n_cls = 2 if head == "binary" else 3
        self.sem = SemanticLoss(self.n_cls, sem_kind, gamma, alpha, alpha_max=alpha_max)
        self.dist = DistanceLoss(dist_kind) if head == "boundary" else None
        self.dist_weight = dist_weight

    @property
    def out_channels(self) -> int:
        return self.n_cls + (1 if self.dist is not None else 0)

    def forward(self, out: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
        target = batch["sem2"] if self.head == "binary" else batch["sem3"]
        l_sem = self.sem(out[:, : self.n_cls], target)
        logs = {"loss_sem": l_sem.item()}
        total = l_sem
        if self.dist is not None:
            l_d = self.dist(out[:, self.n_cls], batch["dist"])
            total = total + self.dist_weight * l_d
            logs["loss_dist"] = l_d.item()
        logs["loss"] = total.item()
        return total, logs
