# Section Writing Guide

Instructions for agents drafting report sections.

---

## Your Task

1. **Read** the outline file in `outlines/`
2. **Write** a full prose draft to `sections/`
3. **Expand** bullet points into complete paragraphs

---

## File Locations

| Section | Outline (input) | Draft (output) |
|---------|-----------------|----------------|
| Introduction | `outlines/01_introduction.md` | `sections/01_introduction.md` |
| Background & Related Work | `outlines/02_background_and_related_work.md` | `sections/02_background_and_related_work.md` |
| Method | `outlines/04_method.md` | `sections/03_method.md` |
| Experiments | `outlines/05_experiments.md` | `sections/04_experiments.md` |
| Discussion | `outlines/06_discussion.md` | `sections/05_discussion.md` |

---

## Key Resources

| Resource | Path | Use For |
|----------|------|---------|
| Project context | `docs/project.md` | Technical details, architecture |
| Research proposal | `docs/proposal.md` | Problem statement, motivation |
| Codebase docs | `docs/codebase.md` | Implementation details |
| Related work | `docs/related-work/resources.md` | Papers to cite |
| Data flow | `docs/data_flow.md` | Dataset, preprocessing |

---

## Writing Style

**Format**: Academic/conference paper style
**Audience**: ML researchers, somewhat familiar with TRM/ARC-AGI
**Tone**: Technical but accessible, no fluff

**Do:**
- Use precise technical language
- Include specific numbers (parameters, accuracies) when available
- Reference figures/tables by number (e.g., "as shown in Figure 1")
- Cite papers using [Author, Year] format
- Keep paragraphs focused (one idea per paragraph)

**Don't:**
- Use emojis
- Use marketing language ("groundbreaking", "revolutionary")
- Repeat information across sections
- Include placeholder text like [TBD] - skip or note what's missing

---

## Section Length Targets

| Section | Target |
|---------|--------|
| Introduction | 1-1.5 pages |
| Background & Related Work | 1.5-2 pages |
| Method | 2-3 pages |
| Experiments | 2-3 pages |
| Discussion | 1.5-2 pages |

Total target: ~10 pages main content

---

## Terminology: Code Names → Report Names

Internal code names must be translated to descriptive names in the report:

| Code Name | Report Name |
|-----------|-------------|
| `standard` | Feedforward Deterministic Encoder |
| `hybrid_standard` | Feedforward Deterministic Encoder (deep) |
| `hybrid_variational` / `hybrid_var` | Cross-Attention VAE |
| `lpn_var` / `lpn_variational` | Per-Demo VAE (LPN-style) |
| `etrmtrm` | Iterative Encoder |

**Other terms to avoid:**
- "semi-final" / "final" experiments → use "preliminary" / "full training"
- "overfit test" → use "preliminary experiments on subset"

See `outlines/04_method.md` Section 3.2 for full architecture descriptions.

---

## Citation Format

Use **slug-based citations**: `[slug]` format.

See `sections/06_references.md` for the full reference list with slugs.

**Example usage:**
> TRM achieves 45% accuracy on ARC-AGI-1 using only 7M parameters [trm].

**Available citation slugs:**

| Slug | Reference |
|------|-----------|
| `[chollet2019]` | Chollet (2019) - ARC benchmark |
| `[trm]` | Jolicoeur-Martineau (2025) - TRM paper |
| `[hrm]` | Wang et al. (2025) - HRM paper |
| `[lpn]` | Bonnet & Macfarlane (2024) - LPN paper |
| `[arc-prize-2024]` | Chollet et al. (2024) - ARC Prize Technical Report |
| `[arc-prize-2025]` | Knoop (2025) - ARC Prize 2025 Results |
| `[hrm-analysis]` | ARC Foundation (2025) - HRM Analysis |
| `[trm-ttt]` | McGovern (2025) - TRM Test-time Training |
| `[trm-inductive]` | (2025) - TRM Inductive Biases paper |
| `[transformer]` | Vaswani et al. (2017) - Transformer |
| `[vae]` | Kingma & Welling (2014) - VAE |
| `[set-transformer]` | Lee et al. (2019) - Set Transformer |
| `[act]` | Graves (2016) - Adaptive Computation Time |
| `[neurosymbolic]` | Chaudhuri et al. (2021) - Neurosymbolic Programming |
| `[induction-transduction]` | Li et al. (2024) - Induction vs Transduction |
| `[hemens-taxonomy]` | Hemens (2025) - ARC taxonomy blog |
| `[nvarc]` | Sorokin & Puget (2025) - NVARC solution |
| `[architects]` | Fiedler et al. (2024) - the ARChitects |

**Note:** Slugs will be converted to numbered citations during final compilation.

---

## Output Format

Write clean markdown with:
- `#` for section title
- `##` for subsections
- `###` for sub-subsections (use sparingly)
- Standard markdown for emphasis, lists, code blocks

---

## Figures, Tables, and Diagrams

**You are NOT expected to create visual assets.** Instead, insert placeholders with brief descriptions. The user will create and fill them in later.

**Placeholder format:**
```
[Figure X: Brief description of what the figure should show]

[Table X: Brief description of what data the table should contain]

[Diagram: Brief description of what the diagram should illustrate]
```

**Examples:**
```
[Figure 1: Side-by-side comparison of TRM (puzzle_id lookup) vs ETRM (encoder) architecture]

[Table 1: Encoder architecture comparison showing type, layers, parameters, and aggregation method]

[Diagram: Data flow showing strict train/eval separation - training demos never include eval puzzles]
```

**Guidelines:**
- Reference placeholders in the text (e.g., "As shown in Figure 1, ...")
- Provide enough description for the user to know what to create
- Suggest what data/comparisons should be included in tables
- Number figures/tables sequentially within each section

---

## Checklist Before Finishing

- [ ] All outline points addressed
- [ ] No [TBD] or placeholder text
- [ ] Citations included where needed
- [ ] Figures/tables referenced (even if not yet created)
- [ ] Length approximately matches target
- [ ] Reads as coherent prose (not bullet points)
