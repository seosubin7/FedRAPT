# -*- coding:utf-8 -*-
"""
Server-side logic for FedRAPT with CCRA (Cross-Client Contrastive Representation Alignment)
"""

import os
import sys
import torch
import copy
import random
import numpy as np
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from federation.client import train, evaluate_client
from models.lstm import LSTM_FedCLIP
from models.contrastive import initialize_prototypes, update_prototypes


class FedCLIP:
    """
    FedRAPT Server with Cross-Client Contrastive Representation Alignment (CCRA)

    Key components:
    - Maintains global class prototypes μ_c for each class c
    - Broadcasts prototypes to clients for contrastive learning
    - Updates prototypes using exponential moving average
    - Aggregates encoder + projection head (FedAvg)
    - Personal classifier heads stay local
    """
    def __init__(self, args):
        self.args = args
        self.nn = LSTM_FedCLIP(args=self.args, name='server').to(args.device)

        # Create client models
        self.nns = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

        # Metrics tracking
        self.history = defaultdict(list)
        self.best_acc = {name: 0.0 for name in self.args.clients}

        # Get shared parameter names (LSTM + projection head)
        self.shared_param_names = list(self.nn.get_shared_params().keys())

        # === CCRA: Class prototypes ===
        projection_dim = getattr(args, 'projection_dim', 64)
        self.prototypes = initialize_prototypes(
            num_classes=args.num_classes,
            embedding_dim=projection_dim
        )
        self.prototype_momentum = getattr(args, 'prototype_beta', 0.9)  # paper's β
        self.proto_update_strategy = getattr(args, 'proto_update_strategy', 'ema')
        self.prototype_counts: dict = {}  # for cumulative average

        # === Communication cost tracking ===
        self.comm_cost = defaultdict(list)
        self._calculate_comm_params()

        print(f"[FedRAPT] Initialized with {args.K} clients")
        print(f"[FedRAPT] Shared params: {len(self.shared_param_names)}")
        print(f"[FedRAPT] Class prototypes: {len(self.prototypes)} classes")
        print(f"[FedRAPT] Communication cost per client per round: {self.params_per_client:,} params (downlink + uplink)")

    def server(self):
        """Main federated learning loop"""
        for t in range(self.args.r):
            print('=' * 80)
            print(f'Round {t + 1}/{self.args.r}')
            print('=' * 80)

            # Sample clients
            selected = self._sample_clients()
            print(f'  Selected clients: {[self.args.clients[idx] for idx in selected]}')

            # Determine whether to use contrastive learning this round
            contrastive_freq = getattr(self.args, 'contrastive_frequency', 1)
            use_contrastive = ((t + 1) % contrastive_freq == 0)

            if use_contrastive:
                print(f'  [Contrastive Learning: ENABLED for this round]')
            else:
                print(f'  [Contrastive Learning: DISABLED (freq={contrastive_freq}, next at round {((t // contrastive_freq) + 1) * contrastive_freq})]')

            # Dispatch: Send encoder + projection + (optionally) prototypes
            self.dispatch(selected, use_contrastive=use_contrastive)

            # Local training: Clients train with CE + (optionally) contrastive loss
            client_embeddings_list = self.client_update(t, selected, use_contrastive=use_contrastive)

            # Update prototypes using client embeddings (only if contrastive learning is enabled)
            if use_contrastive and client_embeddings_list:
                self.prototypes = update_prototypes(
                    current_prototypes=self.prototypes,
                    client_embeddings_list=client_embeddings_list,
                    momentum=self.prototype_momentum,
                    strategy=self.proto_update_strategy,
                    prototype_counts=self.prototype_counts,
                )
                print(f'  [Prototypes updated] Classes: {list(self.prototypes.keys())}')

            # Aggregation: FedAvg on encoder + projection head
            self.aggregation(selected)

            # Log communication cost for this round
            self._log_comm_cost(t + 1, len(selected), use_contrastive=use_contrastive)

            # Evaluation
            if not getattr(self.args, 'no_eval', False):
                self.evaluate(selected, t + 1, scope='personal')

        return self.nn

    def _sample_clients(self):
        """Sample clients for this round"""
        total = self.args.K
        m = getattr(self.args, 'clients_per_round', 0)
        if m is None or m <= 0:
            frac = getattr(self.args, 'C', 1.0)
            m = max(1, min(total, int(round(frac * total))))
        else:
            m = max(1, min(total, m))
        return random.sample(range(total), m)

    def _calculate_comm_params(self):
        """Calculate communication cost parameters"""
        self.shared_params_count = sum(
            p.numel() for name, p in self.nn.named_parameters()
            if name in self.shared_param_names
        )
        self.prototype_params_count = sum(
            proto.numel() for proto in self.prototypes.values()
        )
        self.downlink_params = self.shared_params_count + self.prototype_params_count
        self.uplink_params = self.shared_params_count + self.prototype_params_count
        self.params_per_client = self.downlink_params + self.uplink_params

    def _log_comm_cost(self, round_num, num_clients, use_contrastive=True):
        """Log communication cost for this round"""
        if use_contrastive:
            round_downlink = self.downlink_params * num_clients
            round_uplink = self.uplink_params * num_clients
        else:
            round_downlink = self.shared_params_count * num_clients
            round_uplink = self.shared_params_count * num_clients

        round_total = round_downlink + round_uplink

        self.comm_cost['round'].append(round_num)
        self.comm_cost['num_clients'].append(num_clients)
        self.comm_cost['downlink_params'].append(round_downlink)
        self.comm_cost['uplink_params'].append(round_uplink)
        self.comm_cost['total_params'].append(round_total)
        self.comm_cost['use_contrastive'].append(use_contrastive)

    def dispatch(self, selected, use_contrastive=True):
        """Dispatch global encoder + projection head to selected clients"""
        global_state = self.nn.state_dict()

        for j in selected:
            client_state = self.nns[j].state_dict()
            for name in self.shared_param_names:
                client_state[name] = global_state[name].clone()
            self.nns[j].load_state_dict(client_state)

    def client_update(self, round_idx, selected, use_contrastive=True):
        """Update client models via local training"""
        client_embeddings_list = []

        for k in selected:
            if use_contrastive:
                updated_model, class_embeddings = train(
                    args=self.args,
                    model=self.nns[k],
                    round_idx=round_idx,
                    prototypes=self.prototypes
                )
                client_embeddings_list.append(class_embeddings)
            else:
                updated_model, _ = train(
                    args=self.args,
                    model=self.nns[k],
                    round_idx=round_idx,
                    prototypes=None
                )
            self.nns[k] = updated_model

        return client_embeddings_list

    def aggregation(self, selected):
        """Aggregate shared parameters (encoder + projection head) using FedAvg"""
        s = sum(self.nns[j].len for j in selected)
        if s <= 0:
            return

        global_state = self.nn.state_dict()
        for name in self.shared_param_names:
            global_state[name].zero_()

        for j in selected:
            client_state = self.nns[j].state_dict()
            weight = self.nns[j].len / s
            for name in self.shared_param_names:
                global_state[name] += client_state[name] * weight

        self.nn.load_state_dict(global_state)
        print(f'  [Aggregation] Updated encoder + projection head')

    def global_test(self):
        """Final evaluation on all clients"""
        all_indices = list(range(self.args.K))
        prev_len = len(self.history['round'])
        self.evaluate(all_indices, self.args.r + 1, scope='personal_final')

        new_accs = self.history['accuracy'][prev_len:]
        new_f1 = self.history['f1'][prev_len:]
        new_clients = self.history['client'][prev_len:]
        new_loss = self.history['loss'][prev_len:]

        print('\n' + '=' * 80)
        print('FINAL TEST RESULTS')
        print('=' * 80)
        for c, acc, f1, loss in zip(new_clients, new_accs, new_f1, new_loss):
            print(f"[Final] {c} | Acc {acc:.4f} | F1 {f1:.4f} | Loss {loss:.4f}")

        print(f"\n[Average] Acc {np.mean(new_accs):.4f} | F1 {np.mean(new_f1):.4f}")
        print('=' * 80)

    def evaluate(self, selected, round_number, scope='personal'):
        """Evaluate selected clients on test data"""
        for idx in selected:
            model = self.nns[idx]
            acc, f1, loss = evaluate_client(self.args, model)
            name = self.args.clients[idx]

            prev_best = self.best_acc.get(name, 0.0)
            forgetting = max(0.0, prev_best - acc)
            if acc > prev_best:
                self.best_acc[name] = acc

            self.history['round'].append(round_number)
            self.history['client'].append(name)
            self.history['accuracy'].append(acc)
            self.history['f1'].append(f1)
            self.history['loss'].append(loss)
            self.history['forgetting_rate'].append(forgetting)
            self.history['scope'].append(scope)

            if scope == 'personal':
                print(f'    [{name}] Acc {acc:.4f} | F1 {f1:.4f} | Loss {loss:.4f}')

    def export_history(self):
        """Export training history as DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.history)

    def export_comm_cost(self):
        """Export communication cost as DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.comm_cost)

    def print_comm_summary(self):
        """Print communication cost summary"""
        total_params = sum(self.comm_cost['total_params'])
        total_mb = total_params * 4 / (1024 ** 2)

        print('\n' + '=' * 80)
        print('COMMUNICATION COST SUMMARY')
        print('=' * 80)
        print(f"Shared params (LSTM + Projection): {self.shared_params_count:,}")
        print(f"Prototype params: {self.prototype_params_count:,}")
        print(f"Per client per round: {self.params_per_client:,} params")
        print(f"Total rounds: {len(self.comm_cost['round'])}")
        print(f"Total params transferred: {total_params:,} params")
        print(f"Total size: {total_mb:.2f} MB (float32)")
        print('=' * 80)
