import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


# ============================================================================
# Diversity Loss Functions
# ============================================================================


class DiversityLoss(nn.Module):
    """
    Modular diversity loss functions to prevent latent space collapse in multi-branch reasoning.

    All losses operate on residual components r_m only, not on shared component g.
    Input: list of M residual tensors [r_1, r_2, ..., r_M], each of shape [B, S, H]
    Output: scalar diversity loss
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def none(r_list: list[torch.Tensor]) -> torch.Tensor:
        """Level 0: No diversity loss (baseline)"""
        return torch.tensor(0.0, device=r_list[0].device, dtype=r_list[0].dtype)

    @staticmethod
    def cosine_repulsion(r_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Level 1: Pairwise cosine repulsion

        Encourages residuals to point in different directions.
        L_div = (1 / M(M-1)) * Σ_{m<n} cos²(r_m, r_n)
        """
        if len(r_list) <= 1:
            return torch.tensor(0.0, device=r_list[0].device, dtype=r_list[0].dtype)

        M = len(r_list)
        # Flatten each residual to 1D: [B, S, H] -> [B*S*H]
        # Stack to [M, B*S*H]
        r_flat = torch.stack([r.flatten() for r in r_list], dim=0)  # [M, B*S*H]

        # Normalize
        r_norm = F.normalize(r_flat, dim=1)  # [M, B*S*H]

        # Compute pairwise cosine similarities: (r_m · r_n) / (||r_m|| ||r_n||)
        similarity_matrix = torch.mm(r_norm, r_norm.t())  # [M, M]

        # Sum upper triangle (m < n), square, and average
        mask = torch.triu(torch.ones(M, M, device=similarity_matrix.device), diagonal=1)
        cosine_sq_sum = (similarity_matrix**2 * mask).sum()

        return cosine_sq_sum / (M * (M - 1) / 2)

    @staticmethod
    def orthogonality_loss(r_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Level 1: Matrix orthogonality loss

        Enforces global orthogonality among normalized residuals.
        Z = [r̂_1, ..., r̂_M] (normalized columns)
        L_div = ||Z^T Z - I_M||_F²
        """
        if len(r_list) <= 1:
            return torch.tensor(0.0, device=r_list[0].device, dtype=r_list[0].dtype)

        M = len(r_list)
        # Flatten each to 1D: [B, S, H] -> [B*S*H]
        # Stack to [M, B*S*H]
        r_flat = torch.stack([r.flatten() for r in r_list], dim=0)  # [M, B*S*H]

        # Normalize columns
        r_norm = F.normalize(r_flat, dim=1)  # [M, B*S*H]

        # Compute Gram matrix: R^T R
        gram = torch.mm(r_norm, r_norm.t())  # [M, M]

        # Identity matrix
        identity = torch.eye(M, device=gram.device, dtype=gram.dtype)

        # Frobenius norm: ||Gram - I||_F²
        return ((gram - identity) ** 2).sum()

    @staticmethod
    def repulsion_with_radius(
        r_list: list[torch.Tensor], alpha: float = 1.0, rho: float = 2.0
    ) -> torch.Tensor:
        """
        Level 2: Bounded repulsion with radius constraint

        Combines exponential repulsion with a "stay nearby" constraint.
        L_div = (1 / M(M-1)) * Σ_{m<n} exp(α * cos(r_m, r_n))
        L_radius = (1 / M) * Σ_m max(0, ||r_m - r̄|| - ρ)²
        """
        if len(r_list) <= 1:
            return torch.tensor(0.0, device=r_list[0].device, dtype=r_list[0].dtype)

        M = len(r_list)
        r_flat = torch.stack([r.flatten() for r in r_list], dim=0)  # [M, B*S*H]

        # Repulsion term
        r_norm = F.normalize(r_flat, dim=1)
        similarity_matrix = torch.mm(r_norm, r_norm.t())  # [M, M]
        mask = torch.triu(torch.ones(M, M, device=similarity_matrix.device), diagonal=1)
        repulsion = (torch.exp(alpha * similarity_matrix) * mask).sum() / (
            M * (M - 1) / 2
        )

        # Radius constraint
        r_mean = r_flat.mean(dim=0, keepdim=True)  # [1, B*S*H]
        distances = torch.norm(r_flat - r_mean, dim=1)  # [M]
        radius_penalty = F.relu(distances - rho) ** 2
        radius_loss = radius_penalty.mean()

        return repulsion + radius_loss

    def forward(
        self, r_list: list[torch.Tensor], loss_type: str, **kwargs
    ) -> torch.Tensor:
        """
        Dispatch to appropriate diversity loss function.

        Args:
            r_list: List of M residual tensors, each [B, S, H]
            loss_type: One of: none, cosine_repulsion, orthogonality,
                      repulsion_with_radius, verifier_weighted, dpp
            **kwargs: Additional arguments for specific losses
        """
        if loss_type == "none":
            return self.none(r_list)
        elif loss_type == "cosine_repulsion":
            return self.cosine_repulsion(r_list)
        elif loss_type == "orthogonality":
            return self.orthogonality_loss(r_list)
        elif loss_type == "repulsion_with_radius":
            return self.repulsion_with_radius(
                r_list, alpha=kwargs.get("alpha", 1.0), rho=kwargs.get("rho", 2.0)
            )
        else:
            raise ValueError(f"Unknown diversity loss type: {loss_type}")


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[
        Any,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
        torch.Tensor,
    ]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(
                -1
            )  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (
            self.loss_fn(
                outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
            )
            / loss_divisor
        ).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Diversity losses (for multi-branch reasoning via dimension splitting)
        diversity_loss = outputs.get(
            "diversity_loss", torch.tensor(0.0, device=lm_loss.device)
        )
        branch_magnitude_loss = outputs.get(
            "branch_magnitude_loss", torch.tensor(0.0, device=lm_loss.device)
        )

        # Get loss weights from model config if available
        diversity_weight = (
            getattr(self.model.config, "diversity_loss_weight", 0.0)
            if hasattr(self.model, "config")
            else 0.0
        )
        branch_mag_weight = (
            getattr(self.model.config, "branch_magnitude_weight", 0.0)
            if hasattr(self.model, "config")
            else 0.0
        )

        if diversity_weight > 0:
            metrics["diversity_loss"] = diversity_loss.detach()
        if branch_mag_weight > 0:
            metrics["branch_magnitude_loss"] = branch_magnitude_loss.detach()

        # Track diversity metrics (for monitoring collapse)
        if "pairwise_cosine_sim" in outputs:
            metrics["pairwise_cosine_sim"] = outputs["pairwise_cosine_sim"].detach()
        if "pairwise_l2_dist" in outputs:
            metrics["pairwise_l2_dist"] = outputs["pairwise_l2_dist"].detach()
        if "branch_std" in outputs:
            metrics["branch_std"] = outputs["branch_std"].detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # Total loss combines all components
        total_loss = (
            lm_loss
            + 0.5 * (q_halt_loss + q_continue_loss)
            + diversity_weight * diversity_loss
            + branch_mag_weight * branch_magnitude_loss
        )

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
