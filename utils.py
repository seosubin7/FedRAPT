"""
utils.py — Shared utilities for FedRAPT.
Covers: seed fixing, experiment logging, final model saving.
Note: save_checkpoint/load_checkpoint are provided as utilities but
training resumption from mid-round checkpoints is not yet wired into
the training loop. Only final model weights are saved after training.
"""
import os
import json
import random
import subprocess
import datetime
import numpy as np
import torch


# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------

def set_seed(seed: int):
    """Fix Python / NumPy / PyTorch / CUDA seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------------------------------------
# Experiment logging
# -------------------------------------------------------

def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def save_args_log(args, log_path: str):
    """
    Save the full experiment configuration to a JSON log file.
    Includes all args, run date, and git commit hash.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    def _jsonable(v):
        if isinstance(v, (bool, int, float, str, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        return str(v)

    log = {
        'date':       datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'git_commit': _git_commit_hash(),
        'args':       {k: _jsonable(v) for k, v in sorted(vars(args).items())},
    }
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Config saved: {log_path}")


# -------------------------------------------------------
# Checkpoints
# -------------------------------------------------------

def save_checkpoint(model, optimizer, round_idx: int, path: str, extra: dict = None):
    """
    Save model + optimizer state for resuming.
    extra: any additional metadata (e.g. best_acc, args snapshot).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'round': round_idx,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None):
    """
    Load checkpoint. Returns the saved round index.
    """
    payload = torch.load(path, map_location='cpu')
    model.load_state_dict(payload['model_state'])
    if optimizer is not None and payload.get('optimizer_state') is not None:
        optimizer.load_state_dict(payload['optimizer_state'])
    round_idx = payload.get('round', 0)
    print(f"Checkpoint loaded: {path}  (round {round_idx})")
    return round_idx


def save_final_models(fedclip_server, save_dir: str):
    """Save global model and all personalized client models."""
    os.makedirs(save_dir, exist_ok=True)

    global_path = os.path.join(save_dir, 'global_final.pth')
    torch.save(fedclip_server.nn.state_dict(), global_path)
    print(f"Global model saved: {global_path}")

    for i, name in enumerate(fedclip_server.args.clients):
        client_path = os.path.join(save_dir, f'{name}_final.pth')
        torch.save(fedclip_server.nns[i].state_dict(), client_path)

    print(f"Client models saved: {len(fedclip_server.args.clients)} files in {save_dir}/")
