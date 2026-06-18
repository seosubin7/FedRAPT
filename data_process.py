"""Utility helpers to load client data from per-client NPZ files."""

from __future__ import annotations

import glob
import hashlib
import os
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_DIR = os.environ.get("FEDPER_DATA_DIR", "./data/wisdm_npz")
DEFAULT_PATTERN  = "user_*_windows*.npz"
_LABEL_MAPPING: Dict[str, int] | None = None


def _get_pattern_for_dir(data_dir: str) -> str:
    if 'motionsense' in data_dir.lower():
        return 'subject_*_windows*.npz'
    return 'user_*_windows*.npz'


def _stable_seed(client_name: str) -> int:
    """Deterministic seed from client name using SHA-256 (stable across Python versions)."""
    h = hashlib.sha256(client_name.encode()).hexdigest()
    return int(h[:8], 16)  # 32-bit seed from first 8 hex chars


def _split_idx(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15,
               seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split indices into train/val/test with a fixed seed per client."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset  = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _ensure_label_mapping(data_dir: str) -> Dict[str, int]:
    global _LABEL_MAPPING
    if _LABEL_MAPPING is not None:
        return _LABEL_MAPPING
    pattern = _get_pattern_for_dir(data_dir)
    files   = sorted(glob.glob(os.path.join(data_dir, pattern)))
    labels  = set()
    for fp in files:
        arr = np.load(fp, allow_pickle=True)
        labels.update(str(lbl) for lbl in np.unique(arr['y']))
    _LABEL_MAPPING = {label: idx for idx, label in enumerate(sorted(labels))}
    return _LABEL_MAPPING


def _load_npz(path: str, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NPZ not found: {path}")
    data = np.load(path, allow_pickle=True)
    X    = data["X"].astype(np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected (N, T, C) array, got shape {X.shape}")
    mapping = _ensure_label_mapping(data_dir)
    y_arr   = np.array([mapping[str(lbl)] for lbl in data["y"]], dtype=np.int64)
    return X, y_arr


def nn_seq_wind(client_name: str, batch_size: int, *,
                data_dir: str | None = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_dir = data_dir or DEFAULT_DATA_DIR
    candidates = [
        os.path.join(data_dir, client_name + ".npz"),
        os.path.join(data_dir, client_name + ".NPZ"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        matches = sorted(glob.glob(os.path.join(data_dir, client_name + "*.npz")))
        if not matches:
            raise FileNotFoundError(f"No NPZ found for client '{client_name}' in {data_dir}")
        path = matches[0]

    X, y = _load_npz(path, data_dir)

    # Use SHA-256 based stable seed so train/val/test split is reproducible
    seed = _stable_seed(client_name)
    idx_tr, idx_va, idx_te = _split_idx(len(y), seed=seed)

    Dtr = _to_loader(X[idx_tr], y[idx_tr], batch_size, shuffle=True)
    Val = _to_loader(X[idx_va], y[idx_va], batch_size, shuffle=False)
    Dte = _to_loader(X[idx_te], y[idx_te], batch_size, shuffle=False)
    return Dtr, Val, Dte


def list_clients(data_dir: str | None = None, pattern: str | None = None) -> List[str]:
    data_dir = data_dir or DEFAULT_DATA_DIR
    pattern  = pattern  or DEFAULT_PATTERN
    files    = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")
    return [os.path.splitext(os.path.basename(p))[0] for p in files]


def infer_input_dim(data_dir: str | None = None, pattern: str | None = None) -> int:
    """Return the number of sensor channels C (last axis of X)."""
    data_dir    = data_dir or DEFAULT_DATA_DIR
    clients     = list_clients(data_dir, pattern)
    sample_path = os.path.join(data_dir, clients[0] + ".npz")
    X, _        = _load_npz(sample_path, data_dir)
    return X.shape[-1]   # C, not T*C


def infer_num_classes(data_dir: str | None = None, pattern: str | None = None) -> int:
    data_dir = data_dir or DEFAULT_DATA_DIR
    _ensure_label_mapping(data_dir)
    return len(_LABEL_MAPPING)


__all__ = ["nn_seq_wind", "list_clients", "infer_input_dim", "infer_num_classes"]
