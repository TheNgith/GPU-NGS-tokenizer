"""
Embedding + 2-layer Bidirectional GRU model for trigram-tokenized sequences.

Used as the inference model in both CPU and GPU tokenization pipelines.
Weights are randomly initialized (this is a pipeline integration test,
not a trained model).
"""

import torch
import torch.nn as nn


class TrigramGRU(nn.Module):
    """
    Model architecture:
        1. Embedding layer: maps token IDs (0-95) to dense vectors
        2. 2-layer Bidirectional GRU
        3. Output: final hidden states concatenated (forward + backward)

    Input:  (batch_size, seq_len) int64 tensor of token IDs
    Output: (batch_size, hidden_size * 2) float32 tensor
    """

    def __init__(
        self,
        vocab_size: int = 97,      # 96 trigram codes + 1 padding (idx 0)
        embed_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) int64 token IDs
        Returns:
            (batch_size, hidden_size * 2) float32 — concatenated final
            forward and backward hidden states from the last GRU layer.
        """
        embedded = self.embedding(x)           # (B, L, embed_dim)
        output, hidden = self.gru(embedded)    # hidden: (num_layers*2, B, H)

        # Take the last layer's forward and backward hidden states
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # Last forward:  hidden[-2]
        # Last backward: hidden[-1]
        h_fwd = hidden[-2]   # (B, H)
        h_bwd = hidden[-1]   # (B, H)
        return torch.cat([h_fwd, h_bwd], dim=1)  # (B, H*2)
