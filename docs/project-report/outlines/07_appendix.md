*Cancelled*
# Appendix

## Outline

### Appendix A: Training Diagnostics

Detailed metrics and visualizations supporting the challenges discussed in Section 5.2.

#### A.1 Representation Quality Metrics

**Cross-Sample Variance**:
- Definition: Variance of encoder outputs across different samples in a batch
- Purpose: Detects mode/representation collapse (low variance = collapse)
- Healthy range vs collapsed range

[Figure A1: Cross-sample variance over training steps for each encoder architecture]

[Table A1: Final cross-sample variance values by architecture]

#### A.2 Gradient Flow Analysis

**Encoder Gradient Coverage**:
- Definition: Fraction of samples receiving encoder gradients per batch
- Problem: With caching, only reset samples get gradients
- Before fix: ~2% coverage; After fix: 100% coverage

[Figure A2: Gradient coverage comparison - cached vs re-encoding]

#### A.3 Training Stability

[Figure A3: Loss curves showing training collapse at step ~1900 before gradient clipping]

[Figure A4: Stable training after gradient clipping applied]

---

### Appendix B: Detailed Experiment Results (Optional)

Extended tables with per-puzzle breakdown, additional metrics.

[Table B1: Full preliminary experiment results with all hyperparameters]

[Table B2: Per-architecture training time and memory usage]

---

### Appendix C: Example Predictions (Optional)

Qualitative examples of model predictions.

[Figure C1: Success cases - correctly predicted transformations]

[Figure C2: Failure cases - analysis of common error patterns]

---

*Note: Appendices provide supporting detail. Main findings should be in Discussion.*

## Figures/Tables Summary

| ID | Description | Data Source |
|----|-------------|-------------|
| Figure A1 | Cross-sample variance over training | W&B: `train/cross_sample_var` |
| Figure A2 | Gradient coverage comparison | W&B: `grad/encoder_coverage` or similar |
| Figure A3 | Training collapse (before fix) | Early experiment logs |
| Figure A4 | Stable training (after fix) | Current experiment logs |
| Table A1 | Variance by architecture | W&B summary metrics |
| Table B1 | Full experiment results | W&B runs table |
