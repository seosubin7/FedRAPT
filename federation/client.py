# -*- coding:utf-8 -*-
"""
Client-side training logic for FedRAPT with CCRA
"""
import copy
import os
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.loader import nn_seq_wind
from models.contrastive import prototype_based_infonce_loss, collect_class_embeddings


def get_val_loss(args, model, Val):
    """Compute validation loss"""
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    val_loss = []
    for (seq, label) in Val:
        seq, label = seq.to(args.device), label.to(args.device)
        y_pred = model(seq)
        loss = loss_function(y_pred, label.long())
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)


def train(args, model, round_idx: int, prototypes=None):
    """
    Train model on local data using CE loss + Contrastive loss (CCRA)

    Loss = L_CE + λ * L_CL
    where L_CL uses class prototypes for cross-client alignment.

    Args:
        args: Training arguments
        model: Model to train
        round_idx: Current round index
        prototypes: Dict {class_id: prototype embedding (D,)} from server

    Returns:
        model: Trained model
        class_embeddings: Dict {class_id: mean_embedding (D,)} to send to server
    """
    model.train()
    data_dir = getattr(args, 'data_dir', None)
    Dtr, Val, _ = nn_seq_wind(model.name, args.B, data_dir=data_dir)
    model.len = len(Dtr.dataset) if hasattr(Dtr, 'dataset') else len(Dtr)

    lr = args.lr
    contrastive_weight = getattr(args, 'contrastive_weight', 0.5)
    temperature = getattr(args, 'temperature', 0.07)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=args.weight_decay)

    lr_step = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    total_rounds = getattr(args, 'r', round_idx + 1)
    desc = f"Round {round_idx + 1}/{total_rounds} | {model.name}"
    prog = tqdm(range(args.E), desc=desc, leave=False)
    ce_loss_fn = nn.CrossEntropyLoss().to(args.device)

    use_contrastive = (prototypes is not None and len(prototypes) > 0)

    for epoch in prog:
        train_loss = []
        train_ce_loss = []
        train_cl_loss = []

        for (seq, label) in Dtr:
            seq = seq.to(args.device)
            label = label.to(args.device).long()

            y_pred = model(seq)
            ce_loss = ce_loss_fn(y_pred, label)

            if use_contrastive:
                if getattr(args, 'use_projection', True):
                    z = model.get_contrastive_embedding(seq)
                else:
                    z = model.get_representation(seq)
                    z = torch.nn.functional.normalize(z, p=2, dim=1)

                if getattr(args, 'use_prototypes', True) and len(prototypes) > 0:
                    cl_loss = prototype_based_infonce_loss(
                        embeddings=z,
                        labels=label,
                        prototypes=prototypes,
                        temperature=temperature,
                        use_local_negatives=getattr(args, 'use_local_negatives', True)
                    )
                else:
                    from models.contrastive import local_contrastive_loss
                    cl_loss = local_contrastive_loss(
                        embeddings=z,
                        labels=label,
                        temperature=temperature,
                        use_local_negatives=getattr(args, 'use_local_negatives', True)
                    )
            else:
                cl_loss = torch.tensor(0.0, device=args.device)

            total_loss = ce_loss + contrastive_weight * cl_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss.append(total_loss.cpu().item())
            train_ce_loss.append(ce_loss.cpu().item())
            train_cl_loss.append(cl_loss.cpu().item())

        lr_step.step()

        if use_contrastive:
            tqdm.write(
                '[Round {}/{}][{}] epoch {:03d}/{:03d} '
                'loss {:.6f} (CE {:.6f} + λ*CL {:.6f})'.format(
                    round_idx + 1, total_rounds, model.name,
                    epoch + 1, args.E,
                    np.mean(train_loss), np.mean(train_ce_loss),
                    contrastive_weight * np.mean(train_cl_loss),
                ))
        else:
            tqdm.write(
                '[Round {}/{}][{}] epoch {:03d}/{:03d} '
                'train_loss {:.8f}'.format(
                    round_idx + 1, total_rounds, model.name,
                    epoch + 1, args.E, np.mean(train_loss),
                ))

    best_model = model

    class_embeddings = collect_class_embeddings(
        model=best_model,
        dataloader=Dtr,
        device=args.device,
        num_classes=args.num_classes
    )

    return best_model, class_embeddings


def evaluate_client(args, model):
    """
    Evaluate model on test data

    Returns:
        accuracy, f1, loss
    """
    model.eval()
    data_dir = getattr(args, 'data_dir', None)
    _, _, Dte = nn_seq_wind(model.name, args.B, data_dir=data_dir)

    pred = []
    y = []
    loss_function = nn.CrossEntropyLoss().to(args.device)
    test_loss = []

    with torch.no_grad():
        for (seq, label) in Dte:
            seq = seq.to(args.device)
            label = label.to(args.device).long()

            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            test_loss.append(loss.cpu().item())

            y_pred = torch.argmax(y_pred, dim=-1)
            pred.extend(y_pred.cpu().numpy())
            y.extend(label.cpu().numpy())

    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='weighted', zero_division=0)
    loss = np.mean(test_loss)

    return accuracy, f1, loss
