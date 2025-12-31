"""
Encoder-based TRM (eTRM).

This is a clone of trm.py modified to use a demonstration encoder
instead of learned puzzle embeddings. The encoder computes puzzle
representations from demo examples at runtime.

Key differences from trm.py:
1. No puzzle_emb (learned embedding matrix)
2. _input_embeddings_with_context() accepts external context
3. TRMWithEncoder wrapper combines encoder + TRM inner
4. Carry state caches encoder output for ACT iterations
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import copy
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    LinearSwish,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.encoders import StandardDemoEncoder, DemoEncoderConfig
from models.encoders.lpn_standard import LPNStandardEncoder
from models.encoders.lpn_variational import LPNVariationalEncoder


IGNORE_LABEL_ID = -100


@dataclass
class TRMEncoderInnerCarry:
    """Carry state for TRM inner model (same as original)."""
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TRMEncoderCarry:
    """Carry state for encoder-based TRM with cached context."""
    inner_carry: TRMEncoderInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]
    cached_context: Optional[torch.Tensor] = None  # Cached encoder output


class TRMEncoderConfig(BaseModel):
    """Configuration for encoder-based TRM."""
    batch_size: int
    seq_len: int
    vocab_size: int

    # Encoder settings
    encoder_type: str = "standard"  # "standard", "lpn_standard", "lpn_variational"
    encoder_num_layers: int = 2

    # === Encoder architecture improvements ===
    # Pooling method: "mean", "attention", "weighted"
    encoder_pooling_method: str = "mean"
    # Set encoder layers (cross-attention depth)
    encoder_set_layers: int = 1
    # Layer scale init value (0 = disabled)
    encoder_layer_scale_init: float = 0.0
    # Norm style: "pre" or "post"
    encoder_norm_style: str = "post"
    # QK normalization in attention
    encoder_qk_norm: bool = False

    # TRM reasoning settings
    H_cycles: int
    L_cycles: int
    H_layers: int = 0  # Ignored, kept for compatibility
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Optional settings
    mlp_t: bool = False
    puzzle_emb_len: int = 16  # Context sequence length (from encoder)
    no_ACT_continue: bool = True

    # Diagnostic settings
    freeze_encoder: bool = False  # Freeze encoder weights (for diagnostic experiments)

    # Not used but kept for config compatibility
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 0  # Always 0 for encoder mode


class TRMEncoderBlock(nn.Module):
    """Transformer block for TRM (same as original)."""

    def __init__(self, config: TRMEncoderConfig) -> None:
        super().__init__()
        self.config = config

        if self.config.mlp_t:
            self.puzzle_emb_len = config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
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
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TRMEncoderReasoningModule(nn.Module):
    """Reasoning module that applies input injection (same as original)."""

    def __init__(self, layers: List[TRMEncoderBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TRMEncoderInner(nn.Module):
    """
    TRM inner model modified for encoder mode.

    Key difference: Uses external context instead of puzzle_emb lookup.
    """

    def __init__(self, config: TRMEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O embeddings
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Context length (from encoder)
        self.puzzle_emb_len = config.puzzle_emb_len

        # NO puzzle_emb - using external context from encoder

        # Position encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )

        # Reasoning layers
        self.L_level = TRMEncoderReasoningModule(
            layers=[TRMEncoderBlock(self.config) for _ in range(self.config.L_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings_with_context(
        self, input: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Create input embeddings using external context from encoder.

        Args:
            input: (batch, seq_len) - Input token IDs
            context: (batch, puzzle_emb_len, hidden_size) - Context from encoder

        Returns:
            embeddings: (batch, puzzle_emb_len + seq_len, hidden_size)
        """
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Concatenate context from encoder (replaces puzzle_emb lookup)
        embedding = torch.cat((context, embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device = None) -> TRMEncoderInnerCarry:
        return TRMEncoderInnerCarry(
            z_H=torch.empty(
                batch_size, self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size, dtype=self.forward_dtype, device=device
            ),
            z_L=torch.empty(
                batch_size, self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size, dtype=self.forward_dtype, device=device
            ),
        )

    def reset_carry(
        self, reset_flag: torch.Tensor, carry: TRMEncoderInnerCarry
    ) -> TRMEncoderInnerCarry:
        return TRMEncoderInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: TRMEncoderInnerCarry,
        batch: Dict[str, torch.Tensor],
        context: torch.Tensor,
    ) -> Tuple[TRMEncoderInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with external context.

        Args:
            carry: Inner carry state
            batch: Batch dict with 'inputs', 'labels'
            context: (batch, puzzle_emb_len, hidden_size) - From encoder

        Returns:
            new_carry: Updated carry
            output: Logits (batch, seq_len, vocab_size)
            q_logits: (q_halt, q_continue)
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding with external context
        input_embeddings = self._input_embeddings_with_context(batch["inputs"], context)

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TRMEncoderInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TRMWithEncoder(nn.Module):
    """
    TRM with demonstration encoder.

    Combines StandardDemoEncoder with TRMEncoderInner.
    The encoder computes puzzle representations from demos,
    which are then used as context in the TRM reasoning loop.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMEncoderConfig(**config_dict)

        # Create encoder based on encoder_type
        encoder_config = DemoEncoderConfig(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            num_layers=self.config.encoder_num_layers,
            output_tokens=self.config.puzzle_emb_len,
            vocab_size=self.config.vocab_size,
            seq_len=self.config.seq_len,
            expansion=self.config.expansion,
            rms_norm_eps=self.config.rms_norm_eps,
            forward_dtype=self.config.forward_dtype,
            # Architecture improvements
            pooling_method=self.config.encoder_pooling_method,
            set_encoder_layers=self.config.encoder_set_layers,
            layer_scale_init=self.config.encoder_layer_scale_init,
            norm_style=self.config.encoder_norm_style,
            qk_norm=self.config.encoder_qk_norm,
        )

        if self.config.encoder_type == "standard":
            self.encoder = StandardDemoEncoder(encoder_config)
        elif self.config.encoder_type == "lpn_standard":
            self.encoder = LPNStandardEncoder(encoder_config)
        elif self.config.encoder_type == "lpn_variational":
            self.encoder = LPNVariationalEncoder(encoder_config)
        else:
            raise ValueError(f"Unknown encoder_type: {self.config.encoder_type}")

        # Freeze encoder if requested (for diagnostic experiments)
        if self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Create TRM inner model
        self.inner = TRMEncoderInner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRMEncoderCarry:
        """Initialize carry state."""
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TRMEncoderCarry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),  # Start halted
            current_data={
                k: torch.empty_like(v)
                for k, v in batch.items()
                if k in ["inputs", "labels", "puzzle_identifiers",
                         "demo_inputs", "demo_labels", "demo_mask"]
            },
            cached_context=None,
        )

    def forward(
        self, carry: TRMEncoderCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[TRMEncoderCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass with demo encoding.

        For halted sequences: compute new context from demos.
        For continuing sequences: use cached context.
        """
        # Determine which samples need reset (were halted)
        needs_reset = carry.halted

        # Update current_data for reset samples
        new_current_data = {
            k: torch.where(
                needs_reset.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
            if k in batch
        }

        # Track encoder diagnostics
        encoder_diagnostics = {}

        # Compute context for samples that need reset
        if needs_reset.any():
            # Encode demos for reset samples
            new_context = self.encoder(
                new_current_data["demo_inputs"],
                new_current_data["demo_labels"],
                new_current_data["demo_mask"],
            )

            # Compute encoder diagnostics (detached, no grad impact)
            with torch.no_grad():
                encoder_diagnostics["encoder_output_mean"] = new_context.mean().item()
                encoder_diagnostics["encoder_output_std"] = new_context.std().item()
                encoder_diagnostics["encoder_output_norm"] = new_context.norm(dim=-1).mean().item()
                # Cross-sample variance: how different are encodings across batch items?
                # If this goes to 0, encoder is collapsing
                batch_mean = new_context.mean(dim=0, keepdim=True)  # (1, T, D)
                cross_sample_var = ((new_context - batch_mean) ** 2).mean()
                encoder_diagnostics["encoder_cross_sample_var"] = cross_sample_var.item()
                # Per-token variance across batch
                encoder_diagnostics["encoder_token_std"] = new_context.std(dim=0).mean().item()

            # For continuing samples, use cached context
            if carry.cached_context is not None and not needs_reset.all():
                context = torch.where(
                    needs_reset.view(-1, 1, 1),
                    new_context,
                    carry.cached_context,
                )
            else:
                context = new_context
        else:
            context = carry.cached_context

        # Reset inner carry for halted samples
        new_inner_carry = self.inner.reset_carry(needs_reset, carry.inner_carry)
        new_steps = torch.where(needs_reset, 0, carry.steps)

        # Forward inner model with context
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data, context
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            **encoder_diagnostics,  # Include encoder diagnostics
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # If training and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q for continue
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry, new_current_data, context
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        new_carry = TRMEncoderCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data,
            cached_context=context.detach(),  # Cache for next iteration
        )

        return new_carry, outputs
