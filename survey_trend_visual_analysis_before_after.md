# `survey_trend_visual_analysis.ipynb` Before vs After

This document summarizes the conceptual and implementation changes made in `notebooks/survey_trend_visual_analysis.ipynb`, with explicit formulas for old vs new calculations.

## 1) Scope of Comparison

Comparison baseline:
- **Before**: committed notebook version at `HEAD` before current working-tree edits.
- **After**: current edited notebook in working tree.

Main changed areas:
- Clean divergence heatmaps (core metric logic changed).
- Summary table under clean divergence (metric and formatting changed).
- Public/private terminology and formatting updates.
- Plot export policy and label text consistency.

## 2) Row-Level Data Unit (same before/after)

Each survey row is indexed by:
- `scenario`
- `model`
- `condition` (baseline / persona-reinforcing / alignment-inducing)
- `question_number`, `question_family`
- `participant_role` (alpha/beta)
- `turn`

Public and non-public survey channels are compared per row.

## 3) Terminology and Naming Changes

### Before
- User-facing text often used **private**.
- Scenario title formatting used `.title()` only (so `ngo` became `Ngo`).
- Condition label inconsistency existed in places (`Persona-inducing` vs `Persona-reinforcing`).

### After
- User-facing text uses **OTR** where requested.
- Scenario title helper enforces `Ngo -> NGO`.
- Condition display labels standardized to **Persona-reinforcing**.

Notes:
- Raw input key `survey-private` is intentionally unchanged (source schema dependency).
- Internal analysis columns were renamed from `private_*` / `public_private_*` to `otr_*` / `public_otr_*` where relevant.

## 4) Baseline/Inducing Stacked Bar Metrics

These are the bars with components now labeled:
- `category switch`
- `any score difference`

The arithmetic is unchanged; only naming/column identifiers were updated.

### Per-row definitions
Let:
- `p` = public score
- `o` = OTR score
- `eps` = `ANY_DIFFERENCE_EPSILON` (1e-9)
- `side(x)` in `{negative, neutral, positive}` using epsilon threshold around zero

Indicators:
- `any_diff_row = 1[ |p - o| > eps ]`
- `cat_switch_row = 1[ side(p) != side(o) ]`

### Grouped rates (by role, model, condition)
- `any_difference_rate = 100 * mean(any_diff_row)`
- `category_mismatch_rate = 100 * mean(cat_switch_row)`
- `within_category_difference_rate = max(any_difference_rate - category_mismatch_rate, 0)`

Interpretation:
- Bar total = `any_difference_rate`
- Lower segment = `category_mismatch_rate` (display label: `category switch`)
- Upper segment = `within_category_difference_rate` (display label: `any score difference`)

## 5) Clean Divergence Heatmap: Old vs New

This is the major conceptual change.

---

### Before (old heatmap logic)

Row-level derived columns:
- Signed shift (step-scale percent):
  - `signed_old = 100 * (private - public)`
- Magnitude shift (absolute step-scale percent):
  - `mag_old = 100 * abs(private - public)`

Grouped (by `participant_role`, `condition`, `model`, `scenario`, `question_family`):
- `mean_divergence_pct = mean(mag_old)`
- `signed_private_shift_pct = mean(signed_old)`

Visual encoding:
- Heatmap color used `mean_divergence_pct` (absolute magnitude only).
- Colormap: `YlGnBu`.
- Normalization: `Normalize(vmin=0, vmax=max(20, observed_max))`.
- Sparse in-cell labels showed signed values for high-magnitude cells:
  - show label only when `magnitude >= 20`.

Consequence:
- Colorbar was non-negative (0 to positive max), while text labels could be +/-.

---

### After (new heatmap logic)

Row-level derived column:
- Signed full-range percent:
  - `signed_new = 100 * (otr - public) / 4`
  - denominator `4` is full Likert range width from `-2` to `2`.

Grouped (same dimensions):
- `mean_signed_shift_pct = mean(signed_new)`

Visual encoding:
- Heatmap color uses **signed** grouped value directly (`mean_signed_shift_pct`).
- Colormap: `RdBu_r` (diverging).
- Normalization: `TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=+vmax)` where
  - `vmax = max(5, max_positive, abs(min_negative))`.
- **No in-cell numeric annotations**.

Consequence:
- Color itself now communicates both magnitude and direction.
- Colorbar is symmetric around zero and reflects signed values directly.

## 6) Summary Table Under Clean Divergence: Old vs New

Grouped by:
- `participant_role`
- `condition`
- `question_family`

### Before
- `mean_gap_pct = mean(mag_old)`
- `se_gap_pct = sem(mag_old)`
- Display format: `"mean_gap_pct +/- se_gap_pct %"` (no sign)

### After
- `mean_shift_pct = mean(signed_new)`
- `se_shift_pct = sem(signed_new)`
- Display format: signed mean:
  - `"+x.x +/- y.y%"`

This aligns table values with the new signed heatmap semantics.

## 7) Export and Artifact Policy

### Before
- Global setup had:
  - `SAVE_PLOTS_AS_PDF = False`
  - `SAVE_PLOTS_AS_CSV = True`

### After
- Rerun/export cell enforces:
  - `SAVE_PLOTS_AS_PDF = True`
  - `SAVE_PLOTS_AS_CSV = True`

Clean heatmap artifact stem also changed to signed naming:
- `clean_divergence_signed_shift_scenario_family_heatmaps__{role}`

## 8) What Did Not Change

- Underlying survey row construction granularity.
- Role/model/condition/scenario/question-family grouping structure.
- Bar-chart decomposition arithmetic itself (only naming and OTR refactor changed).

## 9) Practical Verification Checklist

To ensure outputs fully reflect the **after** logic, rerun notebook cells that define and render:
- setup/data-prep cell
- bar metric cell
- clean heatmap cell
- summary table cell
- final rerun/export cell

Old rendered outputs can persist visually until cells are rerun.

