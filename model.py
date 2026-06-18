# -*- coding:utf-8 -*-
"""
FedCLIP Model: LSTM with Contrastive Learning for Personalized FL
"""
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)


class LSTM_FedCLIP(nn.Module):
    """
    LSTM-based model for FedCLIP with contrastive learning

    Architecture:
    - LSTM encoder: shared (aggregated on server)
    - Projection head: shared (for contrastive learning)
    - FC classifier: personalized (kept local)
    """
    def __init__(self, args, name):
        super(LSTM_FedCLIP, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0

        # Model dimensions
        self.input_size = getattr(args, 'input_dim', 3)  # inferred from data
        self.hidden_size = getattr(args, 'hidden_size', 64)
        self.num_layers = 1
        self.num_classes = args.num_classes
        self.projection_dim = getattr(args, 'projection_dim', 64)

        # Ablation study flag
        self.use_personalization = getattr(args, 'use_personalization', True)

        # Shared layer 1: LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Shared layer 2: Projection head for contrastive learning
        # MLP: hidden_size -> 128 -> projection_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.projection_dim)
        )

        # Personalized layers: FC classifier head
        self.fc1 = nn.Linear(self.hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, self.num_classes)

    def forward(self, x):
        """
        Standard forward pass for classification

        Args:
            x: (B, T, F) where T=128, F=3

        Returns:
            logits: (B, num_classes)
        """
        # Handle flattened input for compatibility
        if x.ndim == 2:
            B = x.shape[0]
            x = x.view(B, -1, self.input_size)

        # LSTM forward
        out, _ = self.lstm(x)  # (B, T, hidden_size)
        representation = out[:, -1, :]  # Take last timestep: (B, hidden_size)

        # FC classifier head (personalized)
        out = self.fc1(representation)
        out = self.relu(out)
        logits = self.fc2(out)

        return logits

    def get_representation(self, x):
        """
        Get LSTM representation (before projection head)

        Args:
            x: (B, T, F)

        Returns:
            representation: (B, hidden_size)
        """
        if x.ndim == 2:
            B = x.shape[0]
            x = x.view(B, -1, self.input_size)

        out, _ = self.lstm(x)
        representation = out[:, -1, :]  # (B, hidden_size)

        return representation

    def get_contrastive_embedding(self, x):
        """
        Get contrastive embedding (projected and L2-normalized)

        Args:
            x: (B, T, F)

        Returns:
            embedding: (B, projection_dim) L2-normalized
        """
        representation = self.get_representation(x)

        # Project to contrastive space
        projection = self.projection(representation)  # (B, projection_dim)

        # L2 normalize for cosine similarity
        embedding = F.normalize(projection, p=2, dim=1)

        return embedding

    def get_shared_params(self):
        """
        Return shared (base) parameters

        - If use_personalization=True: LSTM + projection head only
        - If use_personalization=False: All parameters (LSTM + projection + FC)

        These parameters are aggregated on the server.
        """
        shared_params = {}
        for name, param in self.named_parameters():
            if self.use_personalization:
                # Personalized mode: only LSTM + projection are shared
                if name.startswith('lstm.') or name.startswith('projection.'):
                    shared_params[name] = param
            else:
                # Non-personalized mode: all parameters are shared (FedAvg-style)
                shared_params[name] = param
        return shared_params

    def get_personal_params(self):
        """
        Return personalized parameters - FC classifier layers

        - If use_personalization=True: FC classifier layers
        - If use_personalization=False: Empty (no personal params)

        These parameters are kept local and not aggregated.
        """
        personal_params = {}
        if self.use_personalization:
            # Personalized mode: FC layers are personal
            for name, param in self.named_parameters():
                if name.startswith('fc1.') or name.startswith('fc2.') or name.startswith('relu.'):
                    personal_params[name] = param
        # else: no personal params
        return personal_params
