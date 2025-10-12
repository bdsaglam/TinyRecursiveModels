"""
Simplified Tiny Recursive Model (TRM) implementation.

This implementation closely follows the pseudocode from the TRM paper,
making it easier to understand and modify while maintaining full compatibility
with the existing training infrastructure.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import (
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
)
from models.losses import stablemax_cross_entropy
from models.recursive_reasoning.trm_common import TRMBlock, TRMCarry, TRMConfig, TRMNet
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


# ============================================================================
# Configuration
# ============================================================================


class SimpleTRMConfig(TRMConfig):
    pass


# ============================================================================
# Simple TRM Model (following paper pseudocode)
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
            cast_to=self.forward_dtype,
        )

        # Output head
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)

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
                cast_to=self.forward_dtype,
            )

        # Position embeddings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        # ====================================================================
        # The core network (used for both z and y updates)
        # ====================================================================

        self.net = TRMNet(layers=[TRMBlock(config) for _ in range(config.L_layers)])

        # ====================================================================
        # Initial states for y and z
        # ====================================================================

        self.y_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )
        self.z_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )

    def _input_embeddings(
        self, input_tokens: torch.Tensor, puzzle_identifiers: torch.Tensor
    ) -> torch.Tensor:
        """Compute input embeddings with optional puzzle embeddings and position encodings."""
        # Token embeddings
        embedding = self.embed_tokens(input_tokens.to(torch.int32))

        # Add puzzle embeddings if present
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            # Pad if needed
            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size
                - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            # Concatenate puzzle embeddings at the beginning
            embedding = torch.cat(
                (
                    puzzle_embedding.view(
                        -1, self.puzzle_emb_len, self.config.hidden_size
                    ),
                    embedding,
                ),
                dim=-2,
            )

        # Add position embeddings if using learned positions
        if self.config.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        # Scale embeddings
        return self.embed_scale * embedding

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        cos_sin: CosSin,
        n: int = 6,
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
        T: int = 3,
    ) -> Tuple[TRMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        new_carry = TRMCarry(y=y.detach(), z=z.detach())
        lm_output = self.lm_head(y)[:, self.puzzle_emb_len :]

        return new_carry, lm_output

    def init_carry(self, batch_size: int) -> TRMCarry:
        """Initialize carry with learned initial states."""
        return TRMCarry(
            y=self.y_init.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1)
            .clone(),
            z=self.z_init.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, self.config.seq_len + self.puzzle_emb_len, -1)
            .clone(),
        )

    def forward(
        self, carry: TRMCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMCarry, torch.Tensor]:
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
        new_carry, output = self.deep_recursion(
            x=x,
            y=y,
            z=z,
            cos_sin=cos_sin,
            n=self.config.L_cycles,
            T=self.config.H_cycles,
        )

        return new_carry, output


# ============================================================================
# Simple Wrapper (no ACT complexity)
# ============================================================================


class SimpleTRM(nn.Module):
    """
    Simple wrapper around SimpleTRMInner that matches the paper's approach.

    No batch replacement, no complex ACT logic - just fixed supervision steps
    as described in the paper.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = SimpleTRMConfig(**config_dict)
        self.inner = SimpleTRMInner(self.config)

    @property
    def puzzle_emb(self):
        """Expose puzzle embeddings for optimizer access."""
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRMCarry:
        """Create initial carry (y, z) for a new batch."""
        batch_size = batch["inputs"].shape[0]
        return self.inner.init_carry(batch_size)

    def forward(
        self,
        carry: TRMCarry,
        batch: Dict[str, torch.Tensor],
        return_keys: list = [],
    ) -> Tuple[
        TRMCarry, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        Forward pass with loss computation.

        Returns: (new_carry, loss, metrics, outputs)
        """
        # Run one supervision step (deep recursion)
        new_carry, logits = self.inner(carry, batch)

        labels = batch["labels"]

        # Compute loss and metrics
        with torch.no_grad():
            # Predictions
            preds = torch.argmax(logits, dim=-1)

            # Correctness
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics
            batch_size = labels.shape[0]
            metrics = {
                "count": torch.tensor(
                    batch_size, dtype=torch.float32, device=labels.device
                ),
                "accuracy": (is_correct.to(torch.float32) / loss_divisor).sum(-1).sum(),
                "exact_accuracy": seq_is_correct.sum().float(),
            }

        # Loss
        lm_loss = (
            stablemax_cross_entropy(
                logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
            )
            / loss_divisor
        ).sum()

        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
            }
        )

        # Outputs for evaluation
        outputs = {
            k: logits.detach() if k == "logits" else preds.detach()
            for k in return_keys
            if k in ["logits", "preds"]
        }

        return new_carry, lm_loss, metrics, outputs
