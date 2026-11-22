"""Neural Poisson frequency models in PyTorch, with the same fit/predict interface as the sklearn ones.

Two variants:
  NeuralPoisson()               - categorical embeddings + a small MLP, trained on Poisson loss
  NeuralPoisson(offset=glm)     - "CANN" (combined actuarial neural net): the network starts from the
                                  fitted GLM's prediction and only learns a correction to it

Exposure enters as an offset (log exposure), so a 3-month policy is expected to have a quarter of the
claims of a full year; this is the same weighting the GLM and gradient boosting get via sample_weight.

Early stopping leaves the network's overall level slightly off, so after training a single factor rescales
predictions until total predicted claims equal total observed claims on the training data (the "balance
property" a Poisson GLM has by construction). It is computed on training data only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from claims_model import models

NUMERIC = ["veh_power", "veh_age", "driv_age", "bonus_malus", "log_density"]


@dataclass
class Encoding:
    """Category vocabularies and numeric scaling, learned on the training data only."""

    vocab: dict[str, dict[str, int]] = field(default_factory=dict)  # index 0 = unseen category
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def numeric(X: pd.DataFrame) -> pd.DataFrame:
        out = X[["veh_power", "veh_age", "driv_age", "bonus_malus"]].astype(float).copy()
        out["log_density"] = np.log(X["density"].astype(float).clip(lower=1))
        return out

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "Encoding":
        num = cls.numeric(X)
        return cls(
            vocab={c: {v: i + 1 for i, v in enumerate(sorted(X[c].astype(str).unique()))} for c in models.CATEGORICAL},
            mean={c: float(num[c].mean()) for c in NUMERIC},
            std={c: float(num[c].std() or 1.0) for c in NUMERIC},
        )

    def transform(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        cats = np.stack([X[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(dtype=np.int64)
                         for c in models.CATEGORICAL], axis=1)
        num = self.numeric(X)
        nums = np.stack([((num[c] - self.mean[c]) / self.std[c]).to_numpy(dtype=np.float32) for c in NUMERIC], axis=1)
        return torch.from_numpy(cats), torch.from_numpy(nums)


def embedding_dim(n_categories: int) -> int:
    return max(1, min(8, round(1.6 * n_categories ** 0.56)))


class PoissonNet(nn.Module):
    """Embeddings for each categorical, concatenated with scaled numerics, then an MLP giving log-frequency."""

    def __init__(self, cardinalities: list[int], n_numeric: int, hidden: tuple[int, ...] = (32, 16)) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(n + 1, embedding_dim(n + 1)) for n in cardinalities)
        width = sum(e.embedding_dim for e in self.embeddings) + n_numeric
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(width, h), nn.Tanh()]
            width = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(width, 1)

    def forward(self, cats: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        parts = [emb(cats[:, i]) for i, emb in enumerate(self.embeddings)] + [nums]
        return self.out(self.mlp(torch.cat(parts, dim=1))).squeeze(1)


class NeuralPoisson:
    """sklearn-style wrapper: fit(X, y=frequency, sample_weight=exposure); predict(X) -> frequency."""

    def __init__(self, offset: object | None = None, hidden: tuple[int, ...] = (32, 16), epochs: int = 40,
                 batch_size: int = 2048, lr: float = 2e-3, patience: int = 4, val_fraction: float = 0.1,
                 seed: int = 0) -> None:
        self.offset, self.hidden, self.epochs, self.batch_size = offset, hidden, epochs, batch_size
        self.lr, self.patience, self.val_fraction, self.seed = lr, patience, val_fraction, seed
        self.history: list[dict[str, float]] = []

    def _log_offset(self, X: pd.DataFrame) -> torch.Tensor:
        if self.offset is None:
            return torch.full((len(X),), float(np.log(self.base_rate_)))
        return torch.from_numpy(np.log(np.clip(self.offset.predict(X[models.FEATURES]), 1e-6, None)).astype(np.float32))

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray, sample_weight: pd.Series | np.ndarray) -> "NeuralPoisson":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        exposure = np.asarray(sample_weight, dtype=np.float32)
        claims = np.asarray(y, dtype=np.float32) * exposure
        self.base_rate_ = float(claims.sum() / exposure.sum())
        self.encoding_ = Encoding.fit(X)
        cats, nums = self.encoding_.transform(X)
        log_base = self._log_offset(X) + torch.from_numpy(np.log(exposure))
        target = torch.from_numpy(claims)

        self.net_ = PoissonNet([len(self.encoding_.vocab[c]) for c in models.CATEGORICAL], len(NUMERIC), self.hidden)
        # Start exactly at the offset (portfolio mean or GLM): the network only has to learn the correction.
        nn.init.zeros_(self.net_.out.weight)
        nn.init.zeros_(self.net_.out.bias)

        idx = rng.permutation(len(X))
        n_val = int(len(X) * self.val_fraction)
        val, train = idx[:n_val], idx[n_val:]
        loss_fn = nn.PoissonNLLLoss(log_input=True, full=False, reduction="sum")
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)

        def val_loss() -> float:
            self.net_.eval()
            with torch.no_grad():
                pred = self.net_(cats[val], nums[val]) + log_base[val]
                return float(loss_fn(pred, target[val]) / exposure[val].sum())

        best, best_state, stale = val_loss(), copy.deepcopy(self.net_.state_dict()), 0
        self.history = [{"epoch": 0, "val_loss": best}]
        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            for batch in np.array_split(rng.permutation(train), max(1, len(train) // self.batch_size)):
                opt.zero_grad()
                pred = self.net_(cats[batch], nums[batch]) + log_base[batch]
                loss = loss_fn(pred, target[batch]) / len(batch)
                loss.backward()
                opt.step()
            current = val_loss()
            self.history.append({"epoch": epoch, "val_loss": current})
            if current < best - 1e-6:
                best, best_state, stale = current, copy.deepcopy(self.net_.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_epoch_ = min(self.history, key=lambda h: h["val_loss"])["epoch"]
        self.balance_ = 1.0
        self.balance_ = float(claims.sum() / (self.predict(X) * exposure).sum())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cats, nums = self.encoding_.transform(X)
        self.net_.eval()
        with torch.no_grad():
            return self.balance_ * torch.exp(self.net_(cats, nums) + self._log_offset(X)).numpy().astype(float)
