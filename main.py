# main.py
import os
import sys

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import torch
from args import args_parser
from server import FedCLIP
from data_process import infer_input_dim, infer_num_classes
from utils import set_seed, save_args_log, save_final_models


def main():
    args = args_parser()

    # Seed
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    else:
        print("Random seed not set - using random initialization")

    os.environ["FEDPER_DATA_DIR"] = args.data_dir

    args.input_dim   = infer_input_dim(args.data_dir, args.client_pattern)
    args.num_classes = infer_num_classes(args.data_dir, args.client_pattern)

    # Save experiment config log
    log_path = args.metrics_csv.replace('.csv', '_config.json')
    save_args_log(args, log_path)

    print("\n" + "=" * 80)
    print("FedRAPT: Federated Representation Alignment via Prototype-based conTrastive learning")
    print("=" * 80)
    print(f"Dataset         : {args.dataset}  ({args.K} clients, {args.clients_per_round} per round)")
    print(f"Input shape     : (batch, 128, {args.input_dim})  |  Classes: {args.num_classes}")
    print(f"Rounds          : {args.r}  |  Local epochs: {args.E}  |  Batch: {args.B}")
    print(f"LR              : {args.lr}  |  Device: {args.device}")
    print(f"Contrastive     : lambda={args.contrastive_weight}  tau={args.temperature}  proj_dim={args.projection_dim}")
    print(f"Prototype update: strategy={args.proto_update_strategy}  beta={args.prototype_beta}")
    if args.seed is not None:
        print(f"Seed            : {args.seed}")
    print("=" * 80 + "\n")

    fedclip = FedCLIP(args)
    fedclip.server()

    # Save models — separated by checkpoint_tag (dataset variant) and run_id
    checkpoint_tag = args.checkpoint_tag or args.dataset.lower()
    save_dir = os.path.join(CURRENT_DIR, 'checkpoints', checkpoint_tag, f'run{args.run_id}')
    save_final_models(fedclip, save_dir)

    # Final evaluation
    fedclip.global_test()
    fedclip.print_comm_summary()

    # Save metrics CSV
    history_df = fedclip.export_history()
    history_df.to_csv(args.metrics_csv, index=False)
    print(f"\nMetrics saved: {args.metrics_csv}")

    comm_csv = args.metrics_csv.replace('.csv', '_comm.csv')
    comm_df  = fedclip.export_comm_cost()
    comm_df.to_csv(comm_csv, index=False)
    print(f"Comm cost saved: {comm_csv}")


if __name__ == '__main__':
    main()
