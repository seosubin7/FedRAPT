# -*- coding:utf-8 -*-
"""
Contrastive Learning Utilities for FedCLIP (Prototype-based CCRA)

Cross-Client Contrastive Representation Alignment (CCRA):
- Server maintains class prototypes (μ_c)
- Client uses prototypes as cross-client positive/negative samples
- InfoNCE loss aligns representations across clients
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional


def prototype_based_infonce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: Dict[int, torch.Tensor],
    temperature: float = 0.07,
    use_local_negatives: bool = True
) -> torch.Tensor:
    """
    Compute InfoNCE loss using class prototypes for cross-client alignment

    Args:
        embeddings: (N, D) L2-normalized contrastive embeddings from current batch
        labels: (N,) class labels for each embedding
        prototypes: Dict {class_id: (D,) prototype embedding}
        temperature: Temperature τ for softmax
        use_local_negatives: If True, also use within-batch negatives

    Returns:
        loss: Scalar InfoNCE loss

    Strategy:
        - For anchor i with label c:
            - Positive: prototype μ_c (from other clients)
            - Positives (optional): other samples in batch with same label
            - Negatives: prototypes μ_{c'} where c' ≠ c
            - Negatives (optional): other samples in batch with different labels
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Handle empty prototypes
    if len(prototypes) == 0:
        return torch.tensor(0.0, device=device)

    # Ensure embeddings are L2-normalized
    embeddings = F.normalize(embeddings, p=2, dim=1)

    losses = []

    for i in range(batch_size):
        anchor = embeddings[i]  # (D,)
        anchor_label = labels[i].item()

        # === Positive samples ===
        positives = []

        # 1. Prototype of same class (cross-client positive)
        if anchor_label in prototypes:
            proto_pos = prototypes[anchor_label].to(device)
            proto_pos = F.normalize(proto_pos, p=2, dim=0)
            positives.append(proto_pos)

        # 2. Within-batch samples with same label
        if use_local_negatives:
            same_class_mask = (labels == anchor_label)
            same_class_mask[i] = False  # Exclude self
            if same_class_mask.any():
                same_class_embeddings = embeddings[same_class_mask]
                positives.extend(same_class_embeddings)

        if len(positives) == 0:
            # No positive samples available, skip this anchor
            continue

        positives = torch.stack(positives) if len(positives) > 1 else positives[0].unsqueeze(0)

        # === Negative samples ===
        negatives = []

        # 1. Prototypes of different classes
        for c, proto in prototypes.items():
            if c != anchor_label:
                proto_neg = proto.to(device)
                proto_neg = F.normalize(proto_neg, p=2, dim=0)
                negatives.append(proto_neg)

        # 2. Within-batch samples with different labels
        if use_local_negatives:
            diff_class_mask = (labels != anchor_label)
            if diff_class_mask.any():
                diff_class_embeddings = embeddings[diff_class_mask]
                negatives.extend(diff_class_embeddings)

        if len(negatives) == 0:
            # No negative samples, skip
            continue

        negatives = torch.stack(negatives) if len(negatives) > 1 else negatives[0].unsqueeze(0)

        # === Compute InfoNCE loss ===
        # Positive similarities: (num_pos,)
        pos_sim = torch.matmul(positives, anchor) / temperature

        # Negative similarities: (num_neg,)
        neg_sim = torch.matmul(negatives, anchor) / temperature

        # Concatenate all: [pos1, pos2, ..., neg1, neg2, ...]
        all_sim = torch.cat([pos_sim, neg_sim])

        # InfoNCE: -log( Σ exp(pos) / Σ exp(all) )
        # = -log( Σ exp(pos) ) + log( Σ exp(all) )
        # For multiple positives, we average over them
        num_pos = len(pos_sim)
        log_sum_exp_pos = torch.logsumexp(pos_sim, dim=0)
        log_sum_exp_all = torch.logsumexp(all_sim, dim=0)

        loss_i = -(log_sum_exp_pos - np.log(num_pos)) + log_sum_exp_all
        losses.append(loss_i)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    loss = torch.stack(losses).mean()
    return loss


def collect_class_embeddings(
    model,
    dataloader,
    device,
    num_classes: int
) -> Dict[int, torch.Tensor]:
    """
    Collect mean embedding for each class from local data

    This is sent to the server to update global class prototypes.

    Args:
        model: Model with get_contrastive_embedding() method
        dataloader: DataLoader for local data
        device: Device
        num_classes: Number of classes

    Returns:
        class_embeddings: Dict {class_id: mean_embedding (D,)}
    """
    model.eval()

    class_embeddings = defaultdict(list)

    with torch.no_grad():
        for (seq, label) in dataloader:
            seq = seq.to(device)
            label = label.to(device)

            # Get contrastive embeddings
            embeddings = model.get_contrastive_embedding(seq)  # (B, D)

            # Group by class
            for i, lbl in enumerate(label):
                class_id = lbl.item()
                class_embeddings[class_id].append(embeddings[i])

    # Compute mean for each class
    class_means = {}
    for class_id, emb_list in class_embeddings.items():
        if len(emb_list) > 0:
            mean_emb = torch.stack(emb_list).mean(dim=0)  # (D,)
            class_means[class_id] = mean_emb.cpu()

    return class_means


def update_prototypes(
    current_prototypes: Dict[int, torch.Tensor],
    client_embeddings_list: list,
    momentum: float = 0.9,
    strategy: str = 'ema',
    prototype_counts: Optional[Dict[int, int]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Update global class prototypes.

    Strategies:
      - 'ema'          : μ^{t+1} = β*μ^t + (1-β)*z̄   (β = momentum, default 0.9)
      - 'direct'       : μ^{t+1} = z̄
      - 'simple_avg'   : μ^{t+1} = (μ^t + z̄) / 2
      - 'cumulative'   : μ^{t+1} = (n*μ^t + z̄) / (n+1)  (requires prototype_counts)

    Args:
        current_prototypes: Dict {class_id: prototype (D,)}
        client_embeddings_list: List of dicts [{class_id: mean_emb}, ...]
        momentum: β in paper notation (default 0.9). Only used when strategy='ema'.
        strategy: One of 'ema', 'direct', 'simple_avg', 'cumulative'.
        prototype_counts: Dict {class_id: int} tracking update count.
                          Required for strategy='cumulative'; updated in-place.

    Returns:
        updated_prototypes: Dict {class_id: updated_prototype (D,)}
    """
    beta = momentum  # paper's β: weight for old prototype
    updated_prototypes = {}

    # Aggregate client embeddings by class
    class_client_embeddings = defaultdict(list)
    for client_dict in client_embeddings_list:
        for class_id, emb in client_dict.items():
            class_client_embeddings[class_id].append(emb)

    for class_id, emb_list in class_client_embeddings.items():
        new_mean = torch.stack(emb_list).mean(dim=0)

        if class_id not in current_prototypes:
            updated_proto = new_mean
            if strategy == 'cumulative' and prototype_counts is not None:
                prototype_counts[class_id] = 1
        else:
            old_proto = current_prototypes[class_id]
            if strategy == 'ema':
                updated_proto = beta * old_proto + (1 - beta) * new_mean
            elif strategy == 'direct':
                updated_proto = new_mean
            elif strategy == 'simple_avg':
                updated_proto = (old_proto + new_mean) / 2.0
            elif strategy == 'cumulative':
                n = prototype_counts.get(class_id, 1) if prototype_counts is not None else 1
                updated_proto = (n * old_proto + new_mean) / (n + 1)
                if prototype_counts is not None:
                    prototype_counts[class_id] = n + 1
            else:
                raise ValueError(f"Unknown prototype update strategy: {strategy}")

        updated_prototypes[class_id] = updated_proto

    # Keep prototypes for classes not seen in this round
    for class_id, proto in current_prototypes.items():
        if class_id not in updated_prototypes:
            updated_prototypes[class_id] = proto

    return updated_prototypes


def initialize_prototypes(num_classes: int, embedding_dim: int) -> Dict[int, torch.Tensor]:
    """
    Initialize prototypes randomly

    Args:
        num_classes: Number of classes
        embedding_dim: Dimension of embeddings

    Returns:
        prototypes: Dict {class_id: random_embedding (D,)}
    """
    prototypes = {}
    for c in range(num_classes):
        # Random initialization, then normalize
        proto = torch.randn(embedding_dim)
        proto = F.normalize(proto, p=2, dim=0)
        prototypes[c] = proto

    return prototypes


def local_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    use_local_negatives: bool = True
) -> torch.Tensor:
    """
    Contrastive loss using only within-batch samples (no prototypes)

    For ablation study: w/o P (no cross-client prototypes)
    Uses supervised contrastive learning within the batch only.

    Args:
        embeddings: (N, D) L2-normalized contrastive embeddings from current batch
        labels: (N,) class labels for each embedding
        temperature: Temperature τ for softmax
        use_local_negatives: If True, use within-batch negatives

    Returns:
        loss: Scalar contrastive loss
    """
    device = embeddings.device

    if not use_local_negatives:
        return torch.tensor(0.0, device=device)

    batch_size = embeddings.shape[0]
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity matrix: (B, B)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    losses = []
    for i in range(batch_size):
        # Positive mask: same class, exclude self
        pos_mask = (labels == labels[i])
        pos_mask[i] = False

        if not pos_mask.any():
            continue

        # Negative mask: different class
        neg_mask = (labels != labels[i])

        if not neg_mask.any():
            continue

        # Positive similarities
        pos_sim = sim_matrix[i, pos_mask]

        # All similarities (pos + neg)
        all_mask = pos_mask | neg_mask
        all_sim = sim_matrix[i, all_mask]

        # InfoNCE loss for multiple positives
        # log(sum(exp(pos)) / sum(exp(all)))
        log_sum_exp_pos = torch.logsumexp(pos_sim, dim=0)
        log_sum_exp_all = torch.logsumexp(all_sim, dim=0)

        loss_i = -(log_sum_exp_pos - np.log(len(pos_sim))) + log_sum_exp_all
        losses.append(loss_i)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()
