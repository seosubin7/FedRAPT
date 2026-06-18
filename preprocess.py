#!/usr/bin/env python3
"""
Dataset preprocessing for FedRAPT.

Converts raw dataset files into per-client .npz format.
Supports optional Dirichlet re-partitioning for non-IID experiments.

Usage:
    # WISDM (user-based split)
    python preprocess.py --dataset wisdm --data_dir /path/to/WISDM_ar_v1.1_raw.txt --output_dir ./data/wisdm_npz

    # UCI-HAR (user-based split)
    python preprocess.py --dataset ucihar --data_dir /path/to/UCI_HAR_Dataset --output_dir ./data/ucihar_npz

    # MotionSense (user-based split)
    python preprocess.py --dataset motionsense --data_dir /path/to/A_DeviceMotion_data --output_dir ./data/motionsense_npz

    # UCI-HAR with Dirichlet re-partitioning (non-IID)
    python preprocess.py --dataset ucihar --data_dir /path/to/UCI_HAR_Dataset --output_dir ./data/ucihar_npz_alpha01 --dirichlet --alpha 0.1
    python preprocess.py --dataset ucihar --data_dir /path/to/UCI_HAR_Dataset --output_dir ./data/ucihar_npz_alpha05 --dirichlet --alpha 0.5
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ----------------------------------------------
# WISDM
# ----------------------------------------------

WISDM_ACTIVITIES = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
WISDM_LABEL_MAP  = {a: i for i, a in enumerate(WISDM_ACTIVITIES)}
WISDM_WINDOW     = 128
WISDM_STEP       = 64
WISDM_ORIG_HZ    = 20
WISDM_TARGET_HZ  = 50


def _upsample_linear(vals: np.ndarray, orig_hz: int, target_hz: int) -> np.ndarray:
    """Upsample (N, C) signal from orig_hz to target_hz via linear interpolation."""
    n_orig = len(vals)
    n_new = int(round(n_orig * target_hz / orig_hz))
    x_orig = np.arange(n_orig, dtype=np.float64)
    x_new  = np.linspace(0, n_orig - 1, n_new)
    return interp1d(x_orig, vals, axis=0, kind='linear')(x_new).astype(np.float32)


def preprocess_wisdm(raw_txt_path, output_dir):
    """
    Parse WISDM_ar_v1.1_raw.txt and save per-user sliding-window .npz files.
    Raw data is collected at 20 Hz; upsampled to 50 Hz via linear interpolation
    before windowing, so the temporal resolution matches UCI-HAR and MotionSense.
    Format: user_id, activity, timestamp, x, y, z;
    Output: user_{id}_windows.npz with keys X (N,128,3) and y (N,) int64
    """
    print(f"[WISDM] raw={raw_txt_path}  output_dir={output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    with open(raw_txt_path, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';')
            parts = line.split(',')
            if len(parts) != 6:
                continue
            user_id, activity, _ts, x, y, z = parts
            try:
                rows.append({
                    'user': int(user_id),
                    'activity': activity.strip(),
                    'x': float(x), 'y': float(y), 'z': float(z),
                })
            except ValueError:
                continue

    df = pd.DataFrame(rows)
    df = df[df['activity'].isin(WISDM_ACTIVITIES)]

    saved = 0
    for user_id, udf in df.groupby('user'):
        windows_X, windows_y = [], []
        for activity, adf in udf.groupby('activity'):
            vals = adf[['x', 'y', 'z']].values.astype(np.float32)
            # Upsample 20 Hz -> 50 Hz via linear interpolation
            vals = _upsample_linear(vals, WISDM_ORIG_HZ, WISDM_TARGET_HZ)
            # z-score normalize per axis
            for col in range(3):
                mu, sd = vals[:, col].mean(), vals[:, col].std()
                if sd > 1e-8:
                    vals[:, col] = (vals[:, col] - mu) / sd
            # sliding window
            for start in range(0, len(vals) - WISDM_WINDOW + 1, WISDM_STEP):
                windows_X.append(vals[start:start + WISDM_WINDOW])
                windows_y.append(WISDM_LABEL_MAP[activity])

        if not windows_X:
            continue
        X = np.stack(windows_X)               # (N, 128, 3)
        y = np.array(windows_y, dtype=np.int64)
        out = os.path.join(output_dir, f'user_{user_id}_windows.npz')
        np.savez(out, X=X, y=y)
        print(f"  Saved user_{user_id}_windows.npz  ({len(y)} samples)")
        saved += 1

    print(f"Done. {saved} clients saved to {output_dir}/")


# ----------------------------------------------
# UCI-HAR
# ----------------------------------------------

def load_ucihar_signals(data_dir, split):
    signals_dir = os.path.join(data_dir, split, 'Inertial Signals')
    acc_x = np.loadtxt(os.path.join(signals_dir, f'total_acc_x_{split}.txt'))
    acc_y = np.loadtxt(os.path.join(signals_dir, f'total_acc_y_{split}.txt'))
    acc_z = np.loadtxt(os.path.join(signals_dir, f'total_acc_z_{split}.txt'))
    return np.stack([acc_x, acc_y, acc_z], axis=2)  # (N, 128, 3)


def preprocess_ucihar(data_dir, output_dir):
    print(f"[UCI-HAR] data_dir={data_dir}  output_dir={output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    X_train = load_ucihar_signals(data_dir, 'train')
    y_train = np.loadtxt(os.path.join(data_dir, 'train', 'y_train.txt'), dtype=int)
    subj_train = np.loadtxt(os.path.join(data_dir, 'train', 'subject_train.txt'), dtype=int)

    X_test = load_ucihar_signals(data_dir, 'test')
    y_test = np.loadtxt(os.path.join(data_dir, 'test', 'y_test.txt'), dtype=int)
    subj_test = np.loadtxt(os.path.join(data_dir, 'test', 'subject_test.txt'), dtype=int)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    subj_all = np.concatenate([subj_train, subj_test])

    print(f"  Total samples: {len(X_all)}, shape: {X_all.shape}")

    for sid in np.unique(subj_all):
        mask = subj_all == sid
        out = os.path.join(output_dir, f'user_{sid}_windows.npz')
        np.savez(out, X=X_all[mask].astype(np.float32), y=y_all[mask].astype(np.int64))
        print(f"  Saved user_{sid}_windows.npz  ({mask.sum()} samples)")

    print(f"Done. {len(np.unique(subj_all))} files saved to {output_dir}/")


# ----------------------------------------------
# MotionSense
# ----------------------------------------------

MOTIONSENSE_ACTIVITY_MAP = {
    'dws': 0, 'ups': 1, 'wlk': 2, 'jog': 3, 'sit': 4, 'std': 5,
}
WINDOW, STRIDE = 128, 64
EPS = 1e-8


def _zscore(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True)
    std  = np.where(std < EPS, 1.0, std)
    return (arr - mean) / std


def _make_windows(seq, labels):
    N = seq.shape[0]
    if N < WINDOW:
        return None, None
    X, y = [], []
    for s in range(0, N - WINDOW + 1, STRIDE):
        X.append(seq[s:s + WINDOW])
        y.append(np.bincount(labels[s:s + WINDOW]).argmax())
    return np.stack(X), np.array(y)


def _process_motionsense_subject(data_dir, subject_id):
    seqs, lbls = [], []
    for trial_folder in sorted(glob.glob(os.path.join(data_dir, '*'))):
        activity = os.path.basename(trial_folder).split('_')[0]
        if activity not in MOTIONSENSE_ACTIVITY_MAP:
            continue
        subject_file = os.path.join(trial_folder, f'sub_{subject_id}.csv')
        if not os.path.exists(subject_file):
            continue
        df = pd.read_csv(subject_file)
        ua_cols = ['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']
        gv_cols = ['gravity.x', 'gravity.y', 'gravity.z']
        if not all(c in df.columns for c in ua_cols + gv_cols):
            continue
        total_acc = (df[ua_cols].values + df[gv_cols].values).astype(np.float32)
        label = MOTIONSENSE_ACTIVITY_MAP[activity]
        seqs.append(total_acc)
        lbls.append(np.full(len(total_acc), label, dtype=np.int32))

    if not seqs:
        return None, None

    combined        = np.concatenate(seqs, axis=0)
    combined_labels = np.concatenate(lbls, axis=0)
    normalized      = _zscore(combined)
    return _make_windows(normalized, combined_labels)


def preprocess_motionsense(data_dir, output_dir):
    print(f"[MotionSense] data_dir={data_dir}  output_dir={output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    for sid in range(1, 25):
        X, y = _process_motionsense_subject(data_dir, sid)
        if X is None:
            print(f"  Subject {sid}: no data, skipping")
            continue
        out = os.path.join(output_dir, f'subject_{sid}_windows.npz')
        np.savez_compressed(out, X=X, y=y.astype(np.int64))
        print(f"  Saved subject_{sid}_windows.npz  ({len(X)} windows)")
        saved += 1

    print(f"Done. {saved} files saved to {output_dir}/")


# ----------------------------------------------
# Dirichlet re-partitioning (non-IID)
# ----------------------------------------------

def _dirichlet_split(y_all, num_clients, alpha, num_classes):
    class_indices  = {c: np.where(y_all == c)[0] for c in range(1, num_classes + 1)}
    client_indices = [[] for _ in range(num_clients)]

    for c in range(1, num_classes + 1):
        indices = class_indices[c].copy()
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        splits = np.split(indices, (np.cumsum(proportions) * len(indices)).astype(int)[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def dirichlet_repartition(data_dir, output_dir, alpha, num_clients=30, seed=42):
    print(f"[Dirichlet] alpha={alpha}  input={data_dir}  output={output_dir}")
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(data_dir, 'user_*_windows.npz')))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    X_all = np.vstack([np.load(f, allow_pickle=True)['X'] for f in files])
    y_all = np.concatenate([np.load(f, allow_pickle=True)['y'] for f in files])

    # Ensure integer labels starting from 1
    unique_labels = np.unique(y_all)
    if unique_labels.min() == 0:
        y_all = y_all + 1

    print(f"  Total samples: {len(X_all)}, classes: {np.unique(y_all)}")

    num_classes    = len(np.unique(y_all))
    client_indices = _dirichlet_split(y_all, num_clients, alpha, num_classes)

    for i, indices in enumerate(client_indices):
        if not indices:
            continue
        out = os.path.join(output_dir, f'user_{i + 1}_windows.npz')
        np.savez(out, X=X_all[indices], y=y_all[indices])
        print(f"  Saved user_{i + 1}_windows.npz  ({len(indices)} samples)")

    print(f"Done. {num_clients} files saved to {output_dir}/")


# ----------------------------------------------
# Entry point
# ----------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='FedRAPT dataset preprocessing')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['wisdm', 'ucihar', 'motionsense'],
                        help='Dataset to preprocess')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to raw dataset file/directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .npz files')
    parser.add_argument('--dirichlet', action='store_true',
                        help='Apply Dirichlet re-partitioning for non-IID split (UCI-HAR only)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Dirichlet concentration parameter (default: 0.1)')
    parser.add_argument('--num_clients', type=int, default=30,
                        help='Number of clients for Dirichlet split (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    if args.dirichlet:
        if args.dataset != 'ucihar':
            raise ValueError('Dirichlet re-partitioning is only supported for ucihar')
        dirichlet_repartition(args.data_dir, args.output_dir, args.alpha, args.num_clients, args.seed)
    elif args.dataset == 'wisdm':
        preprocess_wisdm(args.data_dir, args.output_dir)
    elif args.dataset == 'ucihar':
        preprocess_ucihar(args.data_dir, args.output_dir)
    elif args.dataset == 'motionsense':
        preprocess_motionsense(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
