"""Discrete action readout for the Baba Is AI joint world model."""

import torch
import torch.nn as nn


class DiscreteActionHead(nn.Module):
    """Predict a fixed sequence of categorical actions from NanoWM features.

    Learned transition queries cross-attend to all final video-transformer
    tokens. For Baba, three queries predict the transitions between four
    frames and each query emits one of five actions.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        num_actions: int = 5,
        num_transitions: int = 3,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.num_actions = num_actions
        self.num_transitions = num_transitions
        self.memory_norm = nn.LayerNorm(input_dim)
        self.memory_projection = nn.Linear(input_dim, model_dim)
        self.transition_queries = nn.Parameter(
            torch.empty(1, num_transitions, model_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.classifier = nn.Linear(model_dim, num_actions)
        nn.init.normal_(self.transition_queries, std=0.02)

    def forward(self, hidden_features: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``[batch, transition, action_class]``.

        Args:
            hidden_features: NanoWM features shaped ``[B, F, P, D]``.
        """
        if hidden_features.ndim != 4:
            raise ValueError(
                "Expected hidden_features [B, F, P, D], got "
                f"{tuple(hidden_features.shape)}"
            )
        batch = hidden_features.shape[0]
        memory = hidden_features.flatten(1, 2)
        memory = self.memory_projection(self.memory_norm(memory))
        queries = self.transition_queries.expand(batch, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.classifier(decoded)
