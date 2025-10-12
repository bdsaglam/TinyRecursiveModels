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
from models.layers import CastedEmbedding, CastedLinear, CosSin, RotaryEmbedding
from models.losses import DiversityLoss
from models.recursive_reasoning.trm_simple import (
    SimpleTRMBlock,
    SimpleTRMCarry,
    SimpleTRMConfig,
    SimpleTRMInnerCarry,
    SimpleTRMNet,
)
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


# ============================================================================
# Configuration
# ============================================================================


class BranchTRMConfig(SimpleTRMConfig):
    # Multi-branch reasoning via clone-perturb-recurse-aggregate
    # z: [B, S, H] is cloned into M branches at runtime
    # Each branch runs latent_recursion independently
    # Diversity loss applied to second half of H (H/2:H are residual dims)
    # First half (0:H/2) allowed to converge (shared understanding)
    num_branches: int = 1  # M parallel reasoning branches (1 = standard TRM)

    # Diversity loss configuration
    diversity_loss_type: str = "none"  # Options: none, cosine_repulsion, orthogonality,
    # repulsion_with_radius, dpp
    diversity_loss_weight: float = 0.0  # λ_div weight for diversity loss
    branch_magnitude_weight: float = 0.0  # λ_mag to prevent branch dims from exploding

    # Hyperparameters for specific diversity losses
    diversity_alpha: float = 1.0  # Temperature for repulsion_with_radius
    diversity_rho: float = 2.0  # Radius bound for repulsion_with_radius
    diversity_epsilon: float = 1e-5  # Stability constant for DPP


# ============================================================================
# Core TRM Model (following paper pseudocode)
# ============================================================================


class BranchTRMInner(nn.Module):
    """
    The core TRM model that implements the recursive reasoning algorithm.

    This follows the paper's pseudocode structure:
    - latent_recursion(x, y, z, n): Updates z n times, then updates y once
    - deep_recursion(x, y, z, n, T): Runs latent_recursion T-1 times without grad,
                                       then once with grad
    """

    def __init__(self, config: BranchTRMConfig):
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

        self.net = SimpleTRMNet(
            layers=[SimpleTRMBlock(config) for _ in range(config.L_layers)]
        )

        # ====================================================================
        # Initial states for y and z
        # ====================================================================

        self.y_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )

        # z_init: [H] - standard initialization
        # Branching is handled at runtime via cloning and perturbation
        self.z_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )

        # ====================================================================
        # Diversity loss module
        # ====================================================================

        self.diversity_loss_fn = DiversityLoss()

        # ====================================================================
        # Q-head initialization (for halting)
        # ====================================================================

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

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

    def _fan_out_latent_vector(self, z: torch.Tensor, M: int) -> torch.Tensor:
        """
        Clone z into M branches with small perturbations for diversity.

        Args:
            z: [B, S, H] latent state
            M: number of branches

        Returns:
            z_branches: [B, M, S, H] with M perturbed copies
        """
        B, S, H = z.shape
        H_half = H // 2

        # Expand to [B, M, S, H]
        z_branches = z.unsqueeze(1).expand(B, M, S, H).clone()  # [B, M, S, H]

        # Apply perturbations to branches 1..M-1 (branch 0 stays unperturbed)
        if M > 1:
            if self.training:
                # Small Gaussian noise on second half (residual dimensions)
                # Only perturb branches 1..M-1
                perturbation = (
                    torch.randn(B, M - 1, S, H - H_half, device=z.device, dtype=z.dtype)
                    * 0.1
                )
                z_branches[:, 1:, :, H_half:] = (
                    z_branches[:, 1:, :, H_half:] + perturbation
                )
            else:
                # At inference: deterministic perturbation
                for m in range(1, M):
                    z_branches[:, m, :, H_half:] = z_branches[:, m, :, H_half:] * (
                        1.0 + 0.1 * m
                    )

        return z_branches

    def _fan_in_latent_vectors(self, z_branches: torch.Tensor) -> torch.Tensor:
        """
        Aggregate M branches back to single z.

        Args:
            z_branches: [B, M, S, H] multi-branch latent states

        Returns:
            z: [B, S, H] aggregated latent state
        """
        # Mean over M dimension
        z_aggregated = z_branches.mean(dim=1)  # [B, S, H]

        return z_aggregated

    def _compute_branch_diversity_loss(self, z_branches: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss on second half of H dimensions only.

        Only penalize similarity in the "residual" dimensions (H/2:H).
        First half (0:H/2) can converge (shared understanding).

        Args:
            z_branches: [B, M, S, H] multi-branch latent states

        Returns:
            diversity_loss: scalar loss value
        """
        if self.config.num_branches <= 1 or self.config.diversity_loss_type == "none":
            return torch.tensor(0.0, device=z_branches.device, dtype=z_branches.dtype)

        B, M, S, H = z_branches.shape
        H_half = H // 2

        # Extract second half from each branch (residual dimensions)
        # [B, M, S, H/2] then convert to list of M tensors [B, S, H/2]
        branch_residuals = [z_branches[:, m, :, H_half:] for m in range(M)]

        # Prepare kwargs for diversity loss
        kwargs = {
            "alpha": self.config.diversity_alpha,
            "rho": self.config.diversity_rho,
            "epsilon": self.config.diversity_epsilon,
        }

        return self.diversity_loss_fn(
            branch_residuals, self.config.diversity_loss_type, **kwargs
        )

    def _compute_branch_magnitude_loss(self, z_branches: torch.Tensor) -> torch.Tensor:
        """
        Regularize branch magnitudes to prevent explosion.

        Penalize large values in second half of dimensions (residual).

        Args:
            z_branches: [B, M, S, H] multi-branch latent states

        Returns:
            magnitude_loss: scalar ||z_residuals||²
        """
        if self.config.num_branches <= 1:
            return torch.tensor(0.0, device=z_branches.device, dtype=z_branches.dtype)

        H_half = self.config.hidden_size // 2

        # Extract second half from all branches: [B, M, S, H/2]
        branch_residuals = z_branches[..., H_half:]

        # L2 penalty on residual dimensions (mean over all dimensions)
        magnitude_loss = (branch_residuals**2).mean()

        return magnitude_loss

    def latent_recursion(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        cos_sin: CosSin,
        n: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Latent recursion.

        z: [B, S, H]

        The network operates on full H dimensions. Diversity is encouraged
        via gradient signals from diversity loss, not explicit branching logic.

        Args:
            x: [B, S, H] input embeddings
            y: [B, S, H] current answer
            z: [B, S, H] latent state with implicit branch structure
            cos_sin: rotary embeddings
            n: number of latent cycles

        Returns:
            y: [B, S, H] updated answer
            z: [B, S, H] updated latent
        """
        # Standard TRM latent recursion - works for both single and multi-branch
        # The implicit branching structure is maintained via diversity loss gradients
        for _ in range(n):
            z = self.net(z, x + y, cos_sin)

        # Update y using z (aggregates information from all branch dims)
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
    ) -> Tuple[
        SimpleTRMInnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """
        Deep recursion with multi-branch exploration via cloning.

        For num_branches > 1:
        1. Clone z into M branches with perturbations
        2. Run latent_recursion independently on each branch
        3. Apply diversity loss to second half of H (residual dimensions)
        4. Aggregate branches back to single z
        5. Update y using aggregated z

        Returns: (new_carry, lm_output, q_logits, auxiliary_losses)
        """

        if self.config.num_branches <= 1:
            # Standard TRM: single trajectory
            with torch.no_grad():
                for _ in range(T - 1):
                    y, z = self.latent_recursion(x, y, z, cos_sin, n)

            # Run 1 cycle with gradients
            y, z = self.latent_recursion(x, y, z, cos_sin, n)

            auxiliary_losses = {
                "diversity_loss": torch.tensor(0.0, device=z.device, dtype=z.dtype),
                "branch_magnitude_loss": torch.tensor(
                    0.0, device=z.device, dtype=z.dtype
                ),
            }

        else:
            # Multi-branch: parallel exploration
            # Key: branches evolve independently through ALL T cycles, aggregate ONCE at end
            M = self.config.num_branches
            B, S, H = z.shape

            # === Initialize M independent branch trajectories ===
            z_branches_flat = self._fan_out_latent_vector(z, M).view(
                B * M, S, H
            )  # [B*M, S, H]

            # Expand x and y for all branches (constant across cycles)
            x_expanded = (
                x.unsqueeze(1).expand(B, M, S, H).reshape(B * M, S, H)
            )  # [B*M, S, H]
            y_expanded = (
                y.unsqueeze(1).expand(B, M, S, H).reshape(B * M, S, H)
            )  # [B*M, S, H]

            # === Run T-1 cycles without gradients ===
            # Each branch evolves independently, no aggregation
            with torch.no_grad():
                for t in range(T - 1):
                    # Run latent recursion on all branches in parallel
                    y_expanded, z_branches_flat = self.latent_recursion(
                        x_expanded, y_expanded, z_branches_flat, cos_sin, n
                    )

            # === Run 1 cycle WITH gradients ===
            # Branches continue their independent evolution
            y_expanded, z_branches_flat = self.latent_recursion(
                x_expanded, y_expanded, z_branches_flat, cos_sin, n
            )
            z_branches = z_branches_flat.view(B, M, S, H)  # [B, M, S, H]

            # === Compute diversity loss while branches are still separate ===
            diversity_loss = self._compute_branch_diversity_loss(z_branches)
            branch_magnitude_loss = self._compute_branch_magnitude_loss(z_branches)

            auxiliary_losses = {
                "diversity_loss": diversity_loss,
                "branch_magnitude_loss": branch_magnitude_loss,
            }

            # === NOW aggregate once for final y update ===
            z = self._fan_in_latent_vectors(z_branches)  # [B, M, S, H] -> [B, S, H]
            y = y_expanded.mean(dim=1)

            # Update y once using aggregated z
            y = self.net(y, z, cos_sin)

        # Compute outputs (same for both single and multi-branch)
        new_carry = SimpleTRMInnerCarry(y=y.detach(), z=z.detach())
        lm_output = self.lm_head(y)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(y[:, 0]).to(torch.float32)

        return (
            new_carry,
            lm_output,
            (q_logits[..., 0], q_logits[..., 1]),
            auxiliary_losses,
        )

    def empty_carry(self, batch_size: int) -> SimpleTRMInnerCarry:
        """Create empty carry for a new batch."""
        y = torch.empty(
            batch_size,
            self.config.seq_len + self.puzzle_emb_len,
            self.config.hidden_size,
            dtype=self.forward_dtype,
        )

        # z is always [B, S, H] regardless of num_branches
        # Branching is implicit in the H dimension
        z = torch.empty(
            batch_size,
            self.config.seq_len + self.puzzle_emb_len,
            self.config.hidden_size,
            dtype=self.forward_dtype,
        )

        return SimpleTRMInnerCarry(y=y, z=z)

    def reset_carry(
        self, reset_flag: torch.Tensor, carry: SimpleTRMInnerCarry
    ) -> SimpleTRMInnerCarry:
        """Reset carry to initial states based on reset_flag."""
        new_y = torch.where(reset_flag.view(-1, 1, 1), self.y_init, carry.y)
        new_z = torch.where(reset_flag.view(-1, 1, 1), self.z_init, carry.z)

        return SimpleTRMInnerCarry(y=new_y, z=new_z)

    def forward(
        self, carry: SimpleTRMInnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[
        SimpleTRMInnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """
        Forward pass: runs deep recursion to improve the answer.

        Automatically handles both single-branch [B,S,H] and multi-branch [B,M,S,H] modes.

        Returns: (new_carry, output, q_logits, auxiliary_losses)
        """
        # Get position encodings
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        # Embed inputs
        x = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Run deep recursion (automatically handles multi-branch if z.ndim == 4)
        new_carry, output, q_logits, auxiliary_losses = self.deep_recursion(
            x=x,
            y=carry.y,
            z=carry.z,
            cos_sin=cos_sin,
            n=self.config.L_cycles,
            T=self.config.H_cycles,
        )

        return new_carry, output, q_logits, auxiliary_losses


# ============================================================================
# ACT Wrapper (Adaptive Computation Time)
# ============================================================================


class BranchTRM(nn.Module):
    """
    Wrapper that adds Adaptive Computation Time (ACT) to the core TRM model.

    ACT allows the model to decide when to halt and move to the next example
    during training, making training more efficient.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = BranchTRMConfig(**config_dict)
        self.inner = BranchTRMInner(self.config)

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
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self, carry: SimpleTRMCarry, batch: Dict[str, torch.Tensor]
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
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # ====================================================================
        # Run inner model
        # ====================================================================

        (
            new_inner_carry,
            logits,
            (q_halt_logits, q_continue_logits),
            auxiliary_losses,
        ) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            # Add diversity losses for tracking and optimization
            "diversity_loss": auxiliary_losses.get("diversity_loss", torch.tensor(0.0)),
            "branch_magnitude_loss": auxiliary_losses.get(
                "branch_magnitude_loss", torch.tensor(0.0)
            ),
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
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q for continue loss (if needed)
                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return SimpleTRMCarry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
