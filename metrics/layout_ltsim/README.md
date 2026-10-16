---
title: Layout Transportation-based Similarity Measure
emoji: 📐
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Transportation-based Similarity Measure

This metric implements LTSim for layout similarity and LTSim-MMD for comparing
two layout collections. It follows the official `v1.0.0-alpha` release of
[`mayu-ot/ltsim`](https://github.com/mayu-ot/ltsim/tree/544b8bccfce4630591589cdec5c5473392190039),
commit `544b8bccfce4630591589cdec5c5473392190039`.

## Usage

```python
import evaluate
import numpy as np

metric = evaluate.load("creative-graphic-design/layout-ltsim")

# Shapes are (batch_size, max_elements, 4), (batch_size, max_elements),
# and (batch_size, max_elements). Boxes use normalized cx, cy, width, height.
bbox1 = np.array([[[0.25, 0.25, 0.2, 0.2]]])
mask1 = np.array([[True]])
labels1 = np.array([[1]])
bbox2 = bbox1.copy()
mask2 = mask1.copy()
labels2 = labels1.copy()

metric.add_batch(
    bbox1=bbox1,
    mask1=mask1,
    labels1=labels1,
    bbox2=bbox2,
    mask2=mask2,
    labels2=labels2,
)
print(metric.compute())
# {"ltsim": 1.0, "ltsim_mmd": nan}
```

`ltsim` is the mean of LTSim for aligned pairs in the two batches. LTSim
combines spatial cost, `1 - (GIoU + 1) / 2`, with a label cost of `0` for an
equal label and `1` otherwise, each weighted by `0.5`. Exact balanced OT uses
uniform element mass, `1 / num_elements`, and `ot.emd`. The final score is
`exp(-OT_cost)` with no additional scaling.

`ltsim_mmd` computes the official collection-level affinity
`exp(-EMD / sigma)`. `sigma` is the median of the strict upper-triangular
real/reference EMD matrix. The tagged official helper averages the upper
triangle including its diagonal for each within-collection term. The metric
does not multiply results by `100`; the paper scales some table entries only
for presentation.

The official dependency specifies POT as `^0.9.1`; this workspace resolves and
tests with POT `0.9.7.post1`.

## Inputs and outputs

Each `bbox` input uses normalized center-x, center-y, width, height boxes.
`mask` removes padded elements, and `labels` supplies one integer class label
per element. The two batches must have the same number of layouts because
`ltsim` follows the official aligned-pair API. Each layout can have a
different number of valid elements.

The returned dictionary contains:

- `ltsim`: mean pairwise LTSim, where `1.0` is an exact match.
- `ltsim_mmd`: official MMD-style collection distance. Lower values indicate
  more similar collections.

## Limitations

- Exact OT and all pairwise collection comparisons are CPU-heavy.
- The official sigma calculation has no defined value for fewer than two
  reference layouts, so `ltsim_mmd` is `nan` for a one-layout collection.
- The official tagged implementation returns cost `0` for identical layout
  dictionaries, including two empty layouts, and cost `1` when exactly one
  layout is empty. This metric preserves that behavior.
- The tagged MMD helper includes diagonal entries in its within-collection
  upper-triangle means, which may differ from diagonal-excluded MMD formulas.

## References

- Otani et al., [LTSim: Layout Transportation-based Similarity Measure for Evaluating Layout Generation](https://arxiv.org/abs/2407.12356), 2024.
- [Official LTSim implementation at the pinned revision](https://github.com/mayu-ot/ltsim/tree/544b8bccfce4630591589cdec5c5473392190039).
- [POT `ot.emd` documentation](https://pythonot.github.io/master/all.html#ot.emd).
