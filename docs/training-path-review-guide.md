# Training Path Review Guide

**Purpose**: How to verify the training pipeline is correctly configured and wired together.

---

## Why This Matters

This codebase has **multiple training modes** that evolved over time. When architectures change or configs are renamed, it's easy to accidentally wire components inconsistently (e.g., using a model trained with fixed ACT steps but expecting dynamic halting).

**Common failure pattern**:
- Job calls script → Script loads config → Config loads arch → Arch specifies model
- If any link is wrong, training runs but with wrong behavior (silent failure!)
- Example: Using `num_act_steps=4` (fixed) instead of dynamic halting → steps stuck at 4

---

## Review Methodology

Follow the **chain of dependencies** from entry point to model implementation.

### Step 1: Start at Job File ✅

**Location**: `jobs-*.txt` files

**What to check**:
```bash
# Verify the command uses:
torchrun ... <TRAINING_SCRIPT>.py --config-name <MAIN_CONFIG>
```

**Example**:
```bash
pretrain_etrm.py --config-name cfg_pretrain_etrm_arc_agi_1
```

**Verify**:
- ✅ Script exists at root level
- ✅ Config name makes sense (not pointing to old/removed config)
- ✅ Override params are correct (encoder_type, exploration_prob, etc.)

**Red Flags**:
- ❌ Script name has `_original`, `_old`, `_backup` suffix
- ❌ Config name references removed approach

---

### Step 2: Check Training Script ✅

**Location**: Root level Python scripts (`pretrain_*.py`)

**What to check**:
```python
@hydra.main(config_path="config", config_name="<CONFIG_NAME>", version_base=None)
def launch(hydra_config: DictConfig):
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    model = create_model(config, ...)
```

**Verify**:
- ✅ `config_name` matches what job file specifies
- ✅ Model creation function name makes sense for the training mode
- ✅ Training loop uses correct forward pass (e.g., `_forward_train_original()` vs `_forward_train_online()`)

**Red Flags**:
- ❌ Config name mismatch with job file
- ❌ Comments reference "fixed steps" or "num_act_steps loop"
- ❌ Training function name doesn't match intended mode

---

### Step 3: Trace Main Config ✅

**Location**: `config/cfg_<name>.yaml`

**What to check**:
```yaml
defaults:
  - arch: <ARCH_NAME>    # Points to config/arch/<ARCH_NAME>.yaml
  - _self_

# Verify these match intended behavior:
arch:
  encoder_type: standard/hybrid_standard/etc
  halt_exploration_prob: 0.5
  halt_max_steps: 16

data_paths: ["data/..."]
evaluators:
  - name: arc@ARC
```

**Verify**:
- ✅ `arch` default points to existing arch config
- ✅ Default encoder_type matches what you expect
- ✅ Data path is correct (encoder-mode vs embedding-mode)
- ✅ ACT parameters are configured (exploration, max_steps)

**Red Flags**:
- ❌ Arch default points to non-existent config
- ❌ Data path is wrong (using embedding-mode data for encoder training)
- ❌ Missing ACT/exploration parameters

---

### Step 4: Check Architecture Config ✅

**Location**: `config/arch/*.yaml`

**What to check**:
```yaml
name: <MODULE_PATH>@<CLASS_NAME>
loss:
  name: losses@ACTLossHead

encoder_type: standard
encoder_num_layers: 2

# Critical: Which ACT mode?
num_act_steps: 1           # Fixed: use N steps per batch
halt_exploration_prob: 0.5 # Dynamic: explore during training
halt_max_steps: 16
```

**Verify**:
- ✅ `name` follows pattern: `recursive_reasoning.<module>@<ClassName>`
- ✅ Module file exists at `models/recursive_reasoning/<module>.py`
- ✅ Class is exported from that module
- ✅ ACT mode matches intent (fixed vs dynamic)

**How to tell ACT mode**:
- **Fixed/Online**: `num_act_steps: 4/8/16` (loops N times per batch)
- **Dynamic/Original**: `num_act_steps: 1` with `halt_exploration_prob` (one forward, carry persists)

**Red Flags**:
- ❌ Module path has `_original` or outdated suffix
- ❌ `num_act_steps > 1` but you expect dynamic halting
- ❌ Missing `halt_exploration_prob` for dynamic mode

---

### Step 5: Verify Model Implementation ✅

**Location**: `models/recursive_reasoning/<module>.py`

**What to check**:
```python
class <ClassName>(nn.Module):
    def forward(self, batch, carry=None):
        if self.training:
            return self._forward_train_<mode>(batch, carry)
        else:
            return self._forward_eval(batch)

    def _forward_train_<mode>(self, batch, carry):
        # Check implementation matches expected mode
```

**Verify**:
- ✅ Class name matches arch config
- ✅ Training forward pass matches intended mode
- ✅ Encoder usage matches mode (cached vs re-encoded)

**Mode-specific checks**:

| Mode | Training Forward | Encoder | Behavior |
|------|-----------------|---------|----------|
| **Fixed/Online** | Loop `num_act_steps` times | Cache after first step | N steps per sample |
| **Dynamic/Original** | One forward, carry persists | Re-encode every step | 2-16 steps dynamic |

**Red Flags**:
- ❌ Training function doesn't match mode name
- ❌ Encoder caching in dynamic mode (should re-encode)
- ❌ Carry not persisted in dynamic mode

---

## Original Project Intent

### Two Main Training Approaches

**Approach 1: Original TRM (Embedding Mode)**
- Script: `pretrain.py`
- Config: `cfg_pretrain_arc_agi_1.yaml`
- Arch: `trm.yaml` → `models/recursive_reasoning/trm.py`
- Uses: Learned puzzle embeddings (one embedding per puzzle)
- Purpose: Baseline reproduction, not few-shot generalization

**Approach 2: ETRM (Encoder Mode)**
- Script: `pretrain_etrm.py`
- Config: `cfg_pretrain_etrm_arc_agi_1.yaml`
- Arch: `etrm.yaml` → `models/recursive_reasoning/etrm.py`
- Uses: Demo encoder (no learned embeddings)
- Purpose: True few-shot generalization to unseen puzzles

### Why Encoder Mode Exists

**Problem with Original TRM**:
- Learned puzzle embeddings for ~400 training puzzles + ~400 eval puzzles
- Eval puzzle embeddings are trained (gradients flow during training)
- Not true generalization - model has seen eval puzzle IDs before

**ETRM Solution**:
- Encode demonstration examples on-the-fly
- No learned embeddings - representation comes from demos
- Can generalize to completely unseen puzzles
- Trade-off: Harder to train, but true few-shot learning

### Encoder Types Within ETRM

| Encoder | Layers | Architecture | Use Case |
|---------|--------|--------------|----------|
| `standard` | 2 | Per-demo transformer → mean aggregation | Baseline |
| `hybrid_standard` | 4 | Per-demo transformer → set transformer → mean | Best performer |
| `hybrid_variational` | 4 | Per-demo transformer → cross-attention | Experimental |
| `lpn_standard` | 6 | LPN-style architecture | Comparison to LPN paper |
| `lpn_variational` | 6 | LPN variational (needs test-time optimization) | Research |

### ACT Halting Modes

**Why Two Modes Exist**:

1. **Fixed/Online** (Approach 2, removed):
   - Loop `num_act_steps` times per batch
   - Simpler training, deterministic steps
   - Problem: Doesn't match original TRM dynamics
   - **Status**: Removed from codebase

2. **Dynamic/Original** (Active):
   - One forward per batch, carry persists across batches
   - Q-head learns when to halt through exploration
   - Matches original TRM paper's truncated BPTT
   - **Status**: This is what we use

**Key Insight**: Dynamic halting is more efficient (easy samples halt early) and matches how TRM was originally trained.

---

## Quick Verification Checklist

Use this when reviewing or after making changes:

- [ ] Job file uses correct script and config
- [ ] Script's Hydra decorator matches config name
- [ ] Main config's `arch` default points to existing arch config
- [ ] Arch config's `name` references correct module@class
- [ ] Module file exists and exports the class
- [ ] Training forward pass matches expected mode
- [ ] ACT behavior is correct (fixed vs dynamic)
- [ ] Data path is correct (encoder-mode vs embedding-mode)
- [ ] Encoder usage matches mode (cached vs re-encoded)
- [ ] No `_original`, `_old`, `_backup` suffixes in active code

---

## Common Pitfalls

### Pitfall 1: Config Mismatch
**Symptom**: Training runs but wrong behavior
**Check**: Job config name ≠ Script's hydra config_name
**Fix**: Update one to match the other

### Pitfall 2: Wrong Arch Default
**Symptom**: Hydra error "arch config not found"
**Check**: Main config's `defaults: - arch: <NAME>` doesn't exist
**Fix**: Create arch config or fix default name

### Pitfall 3: Module Import Error
**Symptom**: "ModuleNotFoundError" or class not found
**Check**: Arch config's `name: path@Class` doesn't match file structure
**Fix**: Update module path or move file to match

### Pitfall 4: Silent Wrong Mode
**Symptom**: ACT steps constant (stuck at 4) instead of varying
**Check**: Using fixed `num_act_steps` instead of dynamic halting
**Fix**: Ensure correct arch config and model implementation

### Pitfall 5: Data Path Wrong
**Symptom**: Encoder not being used, or wrong data loader
**Check**: Using embedding-mode data for encoder training
**Fix**: Point to correct preprocessed data directory

---

## Example Review Session

**Task**: Verify SF1 in `jobs-etrm-semi-final.txt`

**Step 1**: Job file says `pretrain_etrm.py --config-name cfg_pretrain_etrm_arc_agi_1`
**Step 2**: Script has `@hydra.main(config_name="cfg_pretrain_etrm_arc_agi_1")` ✅
**Step 3**: Config has `defaults: - arch: etrm` ✅
**Step 4**: Arch config has `name: recursive_reasoning.etrm@TRMWithEncoder` ✅
**Step 5**: `models/recursive_reasoning/etrm.py` exports `TRMWithEncoder` ✅
**Step 6**: Model uses `_forward_train_original()` with dynamic halting ✅

**Result**: All checks pass, training path is correct.

---

## For Future Agents

When asked to review the training path:

1. **Start with the job file** - this is the entry point
2. **Follow the chain**: job → script → config → arch → model
3. **Verify each link**: file exists, names match, behavior correct
4. **Check for mode mismatches**: fixed vs dynamic, cached vs re-encoded
5. **Document findings**: Create/update verification document

**Ask if confused**:
- Which training mode is intended? (embedding vs encoder)
- Which ACT mode? (fixed vs dynamic)
- Which encoder type? (standard, hybrid, etc.)
- Is this a new experiment or reproducing existing work?
