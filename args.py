# args.py
import argparse
import os
import sys
import torch

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data.loader import list_clients

def args_parser():
    parser = argparse.ArgumentParser()

    # ===== Basic Training Settings =====
    parser.add_argument('--E', type=int, default=3, help='local training epochs')
    parser.add_argument('--r', type=int, default=100, help='number of communication rounds')
    parser.add_argument('--B', type=int, default=50, help='local batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='scheduler decay rate')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ===== Federated Learning Settings =====
    parser.add_argument('--K', type=int, default=10, help='number of total clients (will be overridden by actual client count)')
    parser.add_argument('--clients_per_round', type=int, default=12,
                        help='number of clients sampled each round')
    parser.add_argument('--C', type=float, default=1.0,
                        help='fraction of clients sampled per round (used if clients_per_round <= 0)')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')

    # ===== Model Architecture =====
    parser.add_argument('--input_dim', type=int, default=3, help='input dimension per timestep (3 axes: x,y,z)')
    parser.add_argument('--hidden_size', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--num_classes', type=int, default=6, help='number of activity classes')

    # ===== Contrastive Learning Settings (FedCLIP) =====
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                        help='lambda: weight for contrastive loss in total loss')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='temperature parameter for InfoNCE loss')
    parser.add_argument('--projection_dim', type=int, default=64,
                        help='dimension of projection head output')
    parser.add_argument('--contrastive_frequency', type=int, default=1,
                        help='compute contrastive loss every K rounds')
    parser.add_argument('--num_contrastive_samples', type=int, default=100,
                        help='number of samples per client to collect for contrastive learning')
    # ===== Prototype Update Strategy =====
    parser.add_argument('--proto_update_strategy', type=str, default='ema',
                        choices=['ema', 'direct', 'simple_avg', 'cumulative'],
                        help='prototype update strategy')
    parser.add_argument('--prototype_beta', type=float, default=0.9,
                        help='EMA coefficient β in μ=β·μ+(1-β)·z̄ (only used when proto_update_strategy=ema)')

    parser.add_argument('--use_memory_bank', action='store_true',
                        help='use memory bank to store features from previous rounds')
    parser.add_argument('--memory_bank_size', type=int, default=1000,
                        help='maximum size of memory bank')
    parser.add_argument('--contrastive_lr', type=float, default=0.001,
                        help='learning rate for projection head in contrastive learning')

    # ===== Timing / Efficiency =====
    parser.add_argument('--no_eval', action='store_true',
                        help='skip per-round evaluation (for training time measurement)')

    # ===== Ablation Study Settings =====
    parser.add_argument('--use_prototypes', type=lambda x: x.lower() == 'true', default=True,
                        help='use server prototypes (P) for cross-client alignment')
    parser.add_argument('--use_projection', type=lambda x: x.lower() == 'true', default=True,
                        help='use projection head (H) for contrastive space')
    parser.add_argument('--use_local_negatives', type=lambda x: x.lower() == 'true', default=True,
                        help='use local negatives (L) from within-batch samples')

    # ===== Data Settings =====
    parser.add_argument('--dataset', type=str, default='wisdm',
                        help='dataset name: wisdm, ucihar, motionsense')
    parser.add_argument('--data_dir', type=str, default='',
                        help='directory containing client .npz files (auto-set if empty)')
    parser.add_argument('--client_pattern', type=str, default='user_*_windows*.npz',
                        help='glob pattern for client files in data_dir')
    parser.add_argument('--clients_csv', type=str, default='',
                        help='optional: comma-separated client basenames, e.g., "user_5_windows,user_7_windows"')
    parser.add_argument('--metrics_csv', type=str, default='',
                        help='metrics CSV file path (auto-set to {dataset}/fedrapt_metrics.csv if empty)')
    parser.add_argument('--run_id', type=int, default=1,
                        help='run index used to separate checkpoint directories across independent runs')
    parser.add_argument('--checkpoint_tag', type=str, default='',
                        help='subdirectory name under checkpoints/ (defaults to dataset name)')

    args = parser.parse_args()

    # Auto-set data_dir based on dataset if not provided
    if not args.data_dir:
        default_dirs = {
            'wisdm':       './data/wisdm_npz',
            'ucihar':      './data/ucihar_npz',
            'motionsense': './data/motionsense_npz',
        }
        args.data_dir = default_dirs.get(args.dataset.lower(), f'./data/{args.dataset}_npz')
        print(f"[args] data_dir not specified, using default: {args.data_dir}")

    # Auto-set client_pattern based on dataset
    if args.client_pattern == 'user_*_windows*.npz':  # 기본값인 경우만
        if args.dataset.lower() == 'motionsense':
            args.client_pattern = 'subject_*_windows*.npz'

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(CURRENT_DIR, args.data_dir))

    # Auto-detect clients
    if args.clients_csv.strip():
        args.clients = [c.strip() for c in args.clients_csv.split(',') if c.strip()]
    else:
        args.clients = list_clients(args.data_dir, args.client_pattern)

    args.K = len(args.clients)  # Total number of clients

    # Auto-set metrics_csv path: save in dataset-specific folder
    if not args.metrics_csv:
        dataset_folder = os.path.join(CURRENT_DIR, args.dataset.lower())
        os.makedirs(dataset_folder, exist_ok=True)
        args.metrics_csv = os.path.join(dataset_folder, 'fedrapt_metrics.csv')

    return args
