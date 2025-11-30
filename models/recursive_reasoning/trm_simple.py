"""
Simplified Tiny Recursive Model (TRM) implementation.

This implementation closely follows the pseudocode from the TRM paper,
making it easier to understand and modify while maintaining full compatibility
with the existing training infrastructure.
"""

from typing import Tuple, Dict
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding,
    CosSin, CastedEmbedding, CastedLinear
)
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


# ============================================================================
# Data structures for carrying state across supervision steps
# ============================================================================

@dataclass
class SimpleTRMInnerCarry:
    """Carries the latent states y and z across supervision steps."""
    y: torch.Tensor  # Current answer/solution (called z_H in HRM)
    z: torch.Tensor  # Latent reasoning feature (called z_L in HRM)


@dataclass
class SimpleTRMCarry:
    """Carries all state for ACT (Adaptive Computation Time) wrapper."""
    inner_carry: SimpleTRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


# ============================================================================
# Configuration
# ============================================================================

class SimpleTRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int

    # Recursion parameters (matching paper notation)
    H_cycles: int  # T in paper - number of deep recursion cycles
    L_cycles: int  # n in paper - number of latent recursion cycles

    # Architecture
    L_layers: int       # Number of layers in the network
    H_layers: int       # Ignored (for compatibility)
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

class SimpleTRMBlock(nn.Module):
    """A single transformer block with optional MLP-Mixer style architecture."""

    def __init__(self, config: SimpleTRMConfig):
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
                causal=False
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
                hidden_states + out,
                variance_epsilon=self.norm_eps
            )
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps
            )

        # Channel mixing with post-norm
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

        return hidden_states


class SimpleTRMNet(nn.Module):
    """The core network used for both z and y updates."""

    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: CosSin
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states


# ============================================================================
# Core TRM Model (following paper pseudocode)
# ============================================================================

class SimpleTRMInner(nn.Module):
    """
    The core TRM model that implements the recursive reasoning algorithm.

    This follows the paper's pseudocode structure:
    - latent_recursion(x, y, z, n): Updates z n times, then updates y once
    - deep_recursion(x, y, z, n, T): Runs latent_recursion T-1 times without grad,
                                       then once with grad
    """

    def __init__(self, config: SimpleTRMConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # ====================================================================
        # Input/Output layers
        # ====================================================================

        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Token embeddings
        self.embed_tokens = CastedEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )

        # Output heads
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Puzzle embeddings (optional)
        self.puzzle_emb_len = (
            -(config.puzzle_emb_ndim // -config.hidden_size)
            if config.puzzle_emb_len == 0
            else config.puzzle_emb_len
        )
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype
            )

        # Position embeddings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype
            )

        # ====================================================================
        # The core network (used for both z and y updates)
        # ====================================================================

        self.net = SimpleTRMNet(
            layers=[SimpleTRMBlock(config) for _ in range(config.L_layers)]
        )

        # ====================================================================
        # Initial states for y and z
        # ====================================================================

        self.y_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype),
                std=1
            ),
            persistent=True
        )
        self.z_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype),
                std=1
            ),
            persistent=True
        )

        # ====================================================================
        # Q-head initialization (for halting)
        # ====================================================================

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(
        self,
        input_tokens: torch.Tensor,
        puzzle_identifiers: torch.Tensor
    ) -> torch.Tensor:
        """Compute input embeddings with optional puzzle embeddings and position encodings."""
        # Token embeddings
        embedding = self.embed_tokens(input_tokens.to(torch.int32))

        # Add puzzle embeddings if present
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            # Pad if needed
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            # Concatenate puzzle embeddings at the beginning
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2
            )

        # Add position embeddings if using learned positions
        if self.config.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale embeddings
        return self.embed_scale * embedding

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        cos_sin: CosSin,
        n: int = 6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Latent recursion as described in the paper pseudocode:

        for i in range(n):  # latent reasoning
            z = net(x, y, z)
        y = net(y, z)  # refine output answer
        return y, z
        """
        # Update z n times using x, y, and current z
        for _ in range(n):
            z = self.net(z, x + y, cos_sin)

        # Update y once using current y and z (note: no x here!)
        y = self.net(y, z, cos_sin)

        return y, z

    def deep_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        cos_sin: CosSin,
        n: int = 6,
        T: int = 3
    ) -> Tuple[SimpleTRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Deep recursion as described in the paper pseudocode:

        # Recursing T-1 times to improve y and z (no gradients needed)
        with torch.no_grad():
            for j in range(T - 1):
                y, z = latent_recursion(x, y, z, n)
        # Recursing once to improve y and z
        y, z = latent_recursion(x, y, z, n)
        return (y.detach(), z.detach()), output_head(y), Q_head(y)
        """
        # Run T-1 cycles without gradients
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = self.latent_recursion(x, y, z, cos_sin, n)

        # Run 1 cycle with gradients
        y, z = self.latent_recursion(x, y, z, cos_sin, n)

        # Compute outputs
        new_carry = SimpleTRMInnerCarry(y=y.detach(), z=z.detach())
        lm_output = self.lm_head(y)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(y[:, 0]).to(torch.float32)  # Use first position for Q

        return new_carry, lm_output, (q_logits[..., 0], q_logits[..., 1])

    def empty_carry(self, batch_size: int) -> SimpleTRMInnerCarry:
        """Create empty carry for a new batch."""
        return SimpleTRMInnerCarry(
            y=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype
            ),
            z=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype
            ),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: SimpleTRMInnerCarry
    ) -> SimpleTRMInnerCarry:
        """Reset carry to initial states based on reset_flag."""
        return SimpleTRMInnerCarry(
            y=torch.where(reset_flag.view(-1, 1, 1), self.y_init, carry.y),
            z=torch.where(reset_flag.view(-1, 1, 1), self.z_init, carry.z),
        )

    def forward(
        self,
        carry: SimpleTRMInnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[SimpleTRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: runs deep recursion to improve the answer.

        This is the main entry point called during training/inference.
        """
        # Get position encodings
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        # Embed inputs
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Extract current state
        y, z = carry.y, carry.z

        # Run deep recursion
        new_carry, output, q_logits = self.deep_recursion(
            x=x,
            y=y,
            z=z,
            cos_sin=cos_sin,
            n=self.config.L_cycles,
            T=self.config.H_cycles
        )

        return new_carry, output, q_logits


# ============================================================================
# ACT Wrapper (Adaptive Computation Time)
# ============================================================================

class SimpleTRM(nn.Module):
    """
    Wrapper that adds Adaptive Computation Time (ACT) to the core TRM model.

    ACT allows the model to decide when to halt and move to the next example
    during training, making training more efficient.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = SimpleTRMConfig(**config_dict)
        self.inner = SimpleTRMInner(self.config)

    @property
    def puzzle_emb(self):
        """Expose puzzle embeddings for optimizer access."""
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> SimpleTRMCarry:
        """Create initial carry for a new batch."""
        batch_size = batch["inputs"].shape[0]

        return SimpleTRMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Start halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(
        self,
        carry: SimpleTRMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[SimpleTRMCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass with ACT logic.

        This manages halting, exploration, and multiple supervision steps.
        """
        # ====================================================================
        # Update carry for new data
        # ====================================================================

        # Reset inner carry for halted sequences
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        # Reset steps for halted sequences
        new_steps = torch.where(carry.halted, 0, carry.steps)

        # Update current data for halted sequences
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        # ====================================================================
        # Run inner model
        # ====================================================================

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry,
            new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # ====================================================================
        # ACT halting logic
        # ====================================================================

        with torch.no_grad():
            # Increment step counter
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # During training with ACT enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt if Q-head says to halt
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: randomly force minimum number of steps
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q for continue loss (if needed)
                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry,
                        new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits)
                        )
                    )

        return SimpleTRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
