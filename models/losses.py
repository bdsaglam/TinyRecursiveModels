from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


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
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model forward
        new_carry, outputs = self.model(**model_kwargs)

        # Handle training (online learning) vs eval (adaptive halting)
        if self.model.training:
            # Training mode: single ACT step with online learning
            # Training loop calls this num_act_steps times with backward+optim each
            return self._forward_train_step(new_carry, outputs, model_kwargs["batch"], return_keys)
        else:
            # Eval mode: single step with carry-based halting
            return self._forward_single_step(new_carry, outputs, return_keys)

    def _forward_train_step(
        self,
        new_carry,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str],
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Training forward with single-step outputs (online learning).

        Computes loss for this ACT step. Training loop will call
        backward + optim.step() after each step.
        """
        logits = outputs["logits"]
        q_halt_logits = outputs["q_halt_logits"]
        labels = batch["labels"]
        steps = outputs["steps"]  # Current step number

        # Compute mask
        mask = (labels != IGNORE_LABEL_ID)
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

        # LM loss
        lm_loss = (
            self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
            / loss_divisor
        ).sum()

        # Q-halt loss
        with torch.no_grad():
            is_correct = mask & (torch.argmax(logits, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits,
            seq_is_correct.to(q_halt_logits.dtype),
            reduction="sum"
        )

        # Metrics for this step
        with torch.no_grad():
            valid_metrics = (loss_counts > 0)

            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((q_halt_logits >= 0) == seq_is_correct)).sum(),
                "steps": steps.sum(),  # Total steps across batch
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }

            # Preds
            outputs["preds"] = torch.argmax(logits, dim=-1)

        # Pass through encoder diagnostics
        for key in list(outputs.keys()):
            if key.startswith("encoder_"):
                metrics[key] = outputs[key]

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        total_loss = lm_loss + 0.5 * q_halt_loss
        all_halted = torch.tensor(True, device=labels.device)  # Training always "halted" per step

        return new_carry, total_loss, metrics, detached_outputs, all_halted

    def _forward_single_step(
        self,
        new_carry,
        outputs: Dict[str, torch.Tensor],
        return_keys: Sequence[str],
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Eval forward with single step (original logic).
        """
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Pass through encoder diagnostics
        for key in list(outputs.keys()):
            if key.startswith("encoder_"):
                metrics[key] = outputs[key]

        # Q continue loss (not used in eval, but keep for compatibility)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()

