"""
Quick test script for Branch TRM implementation.

Tests:
1. Basic forward pass (single branch, M=1)
2. Multi-branch forward pass (M=4)
3. Diversity loss computation
4. Shape consistency
"""

import torch
from models.recursive_reasoning.trm_branch import BranchTRMConfig, BranchTRMInner

def test_single_branch():
    """Test baseline (M=1) matches standard TRM behavior."""
    print("\n=== Test 1: Single Branch (M=1) ===")

    config = BranchTRMConfig(
        batch_size=2,
        seq_len=64,
        vocab_size=256,
        num_puzzle_identifiers=100,
        H_cycles=2,
        L_cycles=3,
        L_layers=1,
        H_layers=0,
        hidden_size=128,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,  # No puzzle embeddings
        halt_max_steps=4,
        halt_exploration_prob=0.1,
        num_branches=1,  # Single branch
        diversity_loss_type="none",
    )

    model = BranchTRMInner(config)
    model.eval()

    # Create dummy batch
    batch = {
        "inputs": torch.randint(0, 256, (2, 64)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }

    # Initial carry
    carry = model.empty_carry(batch_size=2)
    carry = model.reset_carry(torch.ones(2, dtype=torch.bool), carry)

    # Forward pass
    new_carry, output, q_logits, aux_losses = model(carry, batch)

    print(f"✓ Output shape: {output.shape}")  # Should be [2, 64, 256]
    print(f"✓ y shape: {new_carry.y.shape}")  # Should be [2, 64, 128]
    print(f"✓ z shape: {new_carry.z.shape}")  # Should be [2, 64, 128]
    print(f"✓ Diversity loss: {aux_losses['diversity_loss'].item():.6f}")
    print(f"✓ Branch magnitude loss: {aux_losses['branch_magnitude_loss'].item():.6f}")

    assert output.shape == (2, 64, 256), f"Expected (2, 64, 256), got {output.shape}"
    assert aux_losses['diversity_loss'].item() == 0.0, "Expected no diversity loss for M=1"

    print("✓ Test 1 passed!")


def test_multi_branch():
    """Test multi-branch (M=4) with diversity loss."""
    print("\n=== Test 2: Multi-Branch (M=4) ===")

    config = BranchTRMConfig(
        batch_size=2,
        seq_len=64,
        vocab_size=256,
        num_puzzle_identifiers=100,
        H_cycles=2,
        L_cycles=3,
        L_layers=1,
        H_layers=0,
        hidden_size=128,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,  # No puzzle embeddings
        halt_max_steps=4,
        halt_exploration_prob=0.1,
        num_branches=4,  # Multi-branch
        diversity_loss_type="cosine_repulsion",
        diversity_loss_weight=0.01,
        branch_magnitude_weight=0.001,
    )

    model = BranchTRMInner(config)
    model.train()  # Training mode for perturbations

    # Create dummy batch
    batch = {
        "inputs": torch.randint(0, 256, (2, 64)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }

    # Initial carry
    carry = model.empty_carry(batch_size=2)
    carry = model.reset_carry(torch.ones(2, dtype=torch.bool), carry)

    # Forward pass
    new_carry, output, q_logits, aux_losses = model(carry, batch)

    print(f"✓ Output shape: {output.shape}")  # Should be [2, 64, 256]
    print(f"✓ y shape: {new_carry.y.shape}")  # Should be [2, 64, 128]
    print(f"✓ z shape: {new_carry.z.shape}")  # Should be [2, 64, 128]
    print(f"✓ Diversity loss: {aux_losses['diversity_loss'].item():.6f}")
    print(f"✓ Branch magnitude loss: {aux_losses['branch_magnitude_loss'].item():.6f}")

    assert output.shape == (2, 64, 256), f"Expected (2, 64, 256), got {output.shape}"
    assert aux_losses['diversity_loss'].item() > 0.0, "Expected non-zero diversity loss for M=4"
    assert aux_losses['branch_magnitude_loss'].item() > 0.0, "Expected non-zero magnitude loss for M=4"

    print("✓ Test 2 passed!")


def test_branch_diversity():
    """Test that branches actually diverge with diversity loss."""
    print("\n=== Test 3: Branch Diversity ===")

    config = BranchTRMConfig(
        batch_size=1,
        seq_len=32,
        vocab_size=256,
        num_puzzle_identifiers=100,
        H_cycles=1,
        L_cycles=2,
        L_layers=1,
        H_layers=0,
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,  # No puzzle embeddings
        halt_max_steps=4,
        halt_exploration_prob=0.1,
        num_branches=3,
        diversity_loss_type="cosine_repulsion",
    )

    model = BranchTRMInner(config)
    model.train()

    # Create a single z vector
    z = torch.randn(1, 32, 64)

    # Clone into branches with perturbations: returns [B, M, S, H]
    z_branches = model._fan_out_latent_vector(z, M=3)
    print(f"✓ z_branches shape: {z_branches.shape}")  # Should be [1, 3, 32, 64]

    # Check that branches differ
    H_half = config.hidden_size // 2

    # First half should be identical (no perturbation) across all branches
    assert torch.allclose(z_branches[:, 0, :, :H_half], z_branches[:, 1, :, :H_half]), \
        "First half should be identical across branches"

    # Second half should differ (perturbation applied)
    diff_1_2 = (z_branches[:, 0, :, H_half:] - z_branches[:, 1, :, H_half:]).abs().mean()
    diff_1_3 = (z_branches[:, 0, :, H_half:] - z_branches[:, 2, :, H_half:]).abs().mean()

    print(f"✓ Difference between branch 0 and 1 (second half): {diff_1_2.item():.6f}")
    print(f"✓ Difference between branch 0 and 2 (second half): {diff_1_3.item():.6f}")

    assert diff_1_2 > 0.01, "Branches should differ in second half"
    assert diff_1_3 > 0.01, "Branches should differ in second half"

    # Test aggregation
    z_agg = model._fan_in_latent_vectors(z_branches)
    print(f"✓ Aggregated z shape: {z_agg.shape}")
    assert z_agg.shape == z.shape, f"Expected {z.shape}, got {z_agg.shape}"

    # Compute diversity loss
    div_loss = model._compute_branch_diversity_loss(z_branches)
    print(f"✓ Diversity loss: {div_loss.item():.6f}")
    assert div_loss.item() > 0.0, "Expected non-zero diversity loss"

    print("✓ Test 3 passed!")


def test_gradient_flow():
    """Test that gradients flow through multi-branch architecture."""
    print("\n=== Test 4: Gradient Flow ===")

    config = BranchTRMConfig(
        batch_size=1,
        seq_len=16,
        vocab_size=256,
        num_puzzle_identifiers=100,
        H_cycles=1,
        L_cycles=1,
        L_layers=1,
        H_layers=0,
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        puzzle_emb_ndim=0,
        puzzle_emb_len=0,  # No puzzle embeddings
        halt_max_steps=4,
        halt_exploration_prob=0.1,
        num_branches=2,
        diversity_loss_type="cosine_repulsion",
        diversity_loss_weight=0.01,
    )

    model = BranchTRMInner(config)
    model.train()

    # Create dummy batch
    batch = {
        "inputs": torch.randint(0, 256, (1, 16)),
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
    }

    # Initial carry
    carry = model.empty_carry(batch_size=1)
    carry = model.reset_carry(torch.ones(1, dtype=torch.bool), carry)

    # Forward pass with gradients
    new_carry, output, q_logits, aux_losses = model(carry, batch)

    # Compute dummy loss
    loss = output.sum() + aux_losses['diversity_loss'] * 0.01
    loss.backward()

    # Check that some parameters have gradients
    has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)

    print(f"✓ Parameters with gradients: {has_grads}/{total_params}")
    assert has_grads > 0, "Expected some parameters to have gradients"

    print("✓ Test 4 passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Branch TRM Implementation")
    print("=" * 60)

    test_single_branch()
    test_multi_branch()
    test_branch_diversity()
    test_gradient_flow()

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
