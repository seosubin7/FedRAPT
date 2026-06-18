# -*- coding:utf-8 -*-
"""
FedRAPT Model: LSTM encoder with projection head and personalized FC classifier
"""
import torch
from torch import nn
import torch.nn.functional as F


class LSTM_FedCLIP(nn.Module):
    """
    LSTM-based model for FedRAPT with contrastive learning

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

        self.input_size = getattr(args, 'input_dim', 3)
        self.hidden_size = getattr(args, 'hidden_size', 64)
        self.num_layers = 1
        self.num_classes = args.num_classes
        self.projection_dim = getattr(args, 'projection_dim', 64)

        self.use_personalization = getattr(args, 'use_personalization', True)

        # Shared: LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Shared: Projection head (MLP: hidden_size -> 128 -> projection_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.projection_dim)
        )

        # Personalized: FC classifier head
        self.fc1 = nn.Linear(self.hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, self.num_classes)

    def forward(self, x):
        """
        Standard forward pass for classification

        Args:
            x: (B, T, F) where T=128, F=input_dim

        Returns:
            logits: (B, num_classes)
        """
        if x.ndim == 2:
            B = x.shape[0]
            x = x.view(B, -1, self.input_size)

        out, _ = self.lstm(x)
        representation = out[:, -1, :]

        out = self.fc1(representation)
        out = self.relu(out)
        logits = self.fc2(out)

        return logits

    def get_representation(self, x):
        """Get LSTM representation (before projection head)"""
        if x.ndim == 2:
            B = x.shape[0]
            x = x.view(B, -1, self.input_size)

        out, _ = self.lstm(x)
        return out[:, -1, :]

    def get_contrastive_embedding(self, x):
        """Get L2-normalized projection embedding for contrastive loss"""
        representation = self.get_representation(x)
        projection = self.projection(representation)
        return F.normalize(projection, p=2, dim=1)

    def get_shared_params(self):
        """Return shared parameters (LSTM + projection head) for FedAvg aggregation"""
        shared_params = {}
        for name, param in self.named_parameters():
            if self.use_personalization:
                if name.startswith('lstm.') or name.startswith('projection.'):
                    shared_params[name] = param
            else:
                shared_params[name] = param
        return shared_params

    def get_personal_params(self):
        """Return personalized parameters (FC classifier) kept local"""
        personal_params = {}
        if self.use_personalization:
            for name, param in self.named_parameters():
                if name.startswith('fc1.') or name.startswith('fc2.') or name.startswith('relu.'):
                    personal_params[name] = param
        return personal_params
