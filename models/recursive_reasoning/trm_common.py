"""
Tiny Recursive Model (TRM) implementation.

This implementation closely follows the pseudocode from the TRM paper,
making it easier to understand and modify while maintaining full compatibility
with the existing training infrastructure.
"""

from dataclasses import dataclass

import torch
from pydantic import BaseModel
from torch import nn

from models.layers import (
    Attention,
    CosSin,
    SwiGLU,
    rms_norm,
)

IGNORE_LABEL_ID = -100


# ============================================================================
# Data structures for carrying state across supervision steps
# ============================================================================


@dataclass
class TRMCarry:
    """Carries the latent states y and z across supervision steps."""

    y: torch.Tensor  # Current answer/solution (called z_H in HRM)
    z: torch.Tensor  # Latent reasoning feature (called z_L in HRM)


# ============================================================================
# Configuration
# ============================================================================


class TRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int

    # Recursion parameters (matching paper notation)
    H_cycles: int  # T in paper - number of deep recursion cycles
    L_cycles: int  # n in paper - number of latent recursion cycles

    # Architecture
    L_layers: int  # Number of layers in the network
    H_layers: int  # Ignored (for compatibility)
    hidden_size: int
    expansion: float
    num_heads: int

    # Position encodings
    pos_encodings: str
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # Puzzle embeddings
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16

    # ACT (Adaptive Computation Time)
    halt_max_steps: int
    halt_exploration_prob: float
    no_ACT_continue: bool = True

    # Data type
    forward_dtype: str = "bfloat16"

    # Architecture variant
    mlp_t: bool = False  # Use MLP on sequence length instead of attention


# ============================================================================
# Building blocks
# ============================================================================


class TRMBlock(nn.Module):
    """A single transformer block with optional MLP-Mixer style architecture."""

    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config

        # Sequence mixing layer (attention or MLP on sequence dimension)
        if config.mlp_t:
            self.puzzle_emb_len = (
                -(config.puzzle_emb_ndim // -config.hidden_size)
                if config.puzzle_emb_len == 0
                else config.puzzle_emb_len
            )
            self.mlp_t = SwiGLU(
                hidden_size=config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )

        # Channel mixing layer
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Sequence mixing with post-norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(
                hidden_states + out, variance_epsilon=self.norm_eps
            )
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states
                + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )

        # Channel mixing with post-norm
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

        return hidden_states


class TRMNet(nn.Module):
    """The core network used for both z and y updates."""

    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: CosSin,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states
