"""Whole-frame Stage-2 polish dataset.

Pairs are loaded from a manifest CSV with columns:
    pair_id, input, output

`input` and `output` are paths relative to `--path-prefix` (the dataset root).
No mask is required — polish targets a global style transform, so crops are
sampled uniformly. Random horizontal flip + 180° rotate as augmentation.

Returns (input_t, target_t) tensors of shape (3, H, W), float in [0, 1].
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_rgb(path: str) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


class PolishStage2Dataset(Dataset):
    def __init__(
        self,
        pairs_df,
        crop_size,
        augment: bool = True,
        path_prefix: str | None = None,
    ):
        self.pairs = pairs_df.reset_index(drop=True)
        self.crop = (crop_size, crop_size) if isinstance(crop_size, int) else tuple(crop_size)
        self.augment = augment
        self.path_prefix = path_prefix

    def __len__(self):
        return len(self.pairs)

    def _resolve(self, p: str) -> str:
        return str(Path(self.path_prefix) / p) if self.path_prefix else p

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        inp = _load_rgb(self._resolve(row["input"]))
        tgt = _load_rgb(self._resolve(row["output"]))

        # Snap input to target dims if the export rescaled.
        h, w = tgt.shape[:2]
        if inp.shape[:2] != (h, w):
            inp = cv2.resize(inp, (w, h), interpolation=cv2.INTER_AREA)

        ch, cw = self.crop
        # Pad if image smaller than crop (rare for polish; targets are
        # full-res Lightroom exports).
        if h < ch or w < cw:
            pad_h = max(0, ch - h)
            pad_w = max(0, cw - w)
            inp = cv2.copyMakeBorder(inp, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            tgt = cv2.copyMakeBorder(tgt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            h, w = tgt.shape[:2]

        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        inp = inp[top:top + ch, left:left + cw]
        tgt = tgt[top:top + ch, left:left + cw]

        if self.augment:
            if random.random() > 0.5:
                inp = np.fliplr(inp).copy()
                tgt = np.fliplr(tgt).copy()
            if random.random() > 0.5:
                inp = np.rot90(inp, 2).copy()
                tgt = np.rot90(tgt, 2).copy()

        inp_t = torch.from_numpy(inp.astype(np.float32)).permute(2, 0, 1) / 255.0
        tgt_t = torch.from_numpy(tgt.astype(np.float32)).permute(2, 0, 1) / 255.0
        return inp_t, tgt_t
