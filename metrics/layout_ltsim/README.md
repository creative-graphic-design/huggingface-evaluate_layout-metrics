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

## Description

This metric implements pairwise LTSim and collection-level LTSim-MMD from Otani et al., following the official `mayu-ot/ltsim` release `v1.0.0-alpha` at commit `544b8bccfce4630591589cdec5c5473392190039`.

For each element pair, LTSim combines spatial cost `1 - (GIoU + 1) / 2` and binary category cost with equal weights, solves exact balanced optimal transport with uniform element mass using POT, and returns `exp(-OT_cost)`.

LTSim-MMD converts the official EMD values to affinity with `exp(-EMD / sigma)`, where `sigma` is the median of the strict upper triangle of the first collection's within-collection EMD matrix; each within-collection mean uses the upper triangle including its diagonal.

## Usage

```python
import evaluate

metric = evaluate.load("creative-graphic-design/layout-ltsim")
layouts1 = [{"bboxes": [[0.25, 0.25, 0.2, 0.2]], "categories": [1]}]
layouts2 = [{"bboxes": [[0.25, 0.25, 0.2, 0.2]], "categories": [1]}]
metric.add_batch(layouts1=layouts1, layouts2=layouts2)
print(metric.compute())
# {"ltsim": 1.0, "ltsim_mmd": nan}
```

## Parameters

- `layouts1` (`list` of `dict`): First layout collection; each dictionary contains `bboxes` and `categories`.
- `layouts2` (`list` of `dict`): Second layout collection with the same structure as `layouts1`.
- `bboxes` (`list` of `list` of `float`): Normalized center-x, center-y, width, height boxes in `[0, 1]` coordinates.
- `categories` (`list` of `int`): One integer category label per box.

The collections must have equal length for this metric's aligned pairwise output; each layout may contain a different number of elements. The underlying LTSim-MMD calculation supports asymmetric collection lengths, as covered by the parity tests.

## Returns

The metric returns a dictionary with `ltsim`, the mean aligned-pair similarity where `1.0` is an exact match, and `ltsim_mmd`, the official MMD-style collection distance where lower values indicate more similar collections.

## Interpretation

Higher `ltsim` means more similar aligned layouts. `ltsim_mmd` is a distance-like quantity computed from the official affinity kernel and should be compared only with the same sigma and implementation variant.

The metric returns unscaled values; the paper multiplies some displayed LTSim-MMD table entries by 100 for presentation.

## Limitations

- Exact EMD and collection MMD require many CPU-bound pairwise optimal-transport computations.
- The library implementation followed here returns cost `0` for identical layout dictionaries, including two empty layouts, and cost `1` when exactly one layout is empty; see [`src/layout_eval/measures/ltsim.py`](https://github.com/mayu-ot/ltsim/blob/544b8bccfce4630591589cdec5c5473392190039/src/layout_eval/measures/ltsim.py#L53-L65).
- The official Dataflow implementation used for the paper's collection numbers checks emptiness first and returns cost `0` when either layout is empty; see [`dataflow/src/comp_pair/ltsim.py`](https://github.com/mayu-ot/ltsim/blob/544b8bccfce4630591589cdec5c5473392190039/dataflow/src/comp_pair/ltsim.py#L52-L59). Therefore paper LTSim-MMD table values are not directly comparable when empty layouts occur.
- The official one-layout reference collection has no strict-upper-triangle bandwidth and returns `nan`; this metric also returns `nan` for an identical reference collection whose sigma is zero, without emitting a `RuntimeWarning`.
- The tagged MMD helper includes diagonal entries in its within-collection means, which differs from diagonal-excluded MMD definitions.
- The Hugging Face Space must be created by the repository owner before the main-branch deploy can publish it; deployment also requires the repository's existing Hub credentials.

## Citations

```bibtex
@article{otani2024ltsim,
  title={LTSim: Layout Transportation-based Similarity Measure for Evaluating Layout Generation},
  author={Otani, Mayu and Inoue, Naoto and Kikuchi, Kotaro and Togashi, Ryo},
  journal={arXiv preprint arXiv:2407.12356},
  year={2024}
}
```

- [Paper](https://arxiv.org/abs/2407.12356)
- [Pinned official implementation](https://github.com/mayu-ot/ltsim/tree/544b8bccfce4630591589cdec5c5473392190039)
- [POT `ot.emd` documentation](https://pythonot.github.io/master/all.html#ot.emd)
