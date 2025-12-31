---
title: Layout Overlap
emoji: 📊
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.17.0
app_file: app.py
pinned: false
---

# Layout Overlap

## Description

The Layout Overlap metric quantifies how much layout elements overlap with each other. This metric implements three different overlap calculation methods from influential layout generation research papers, providing comprehensive evaluation of element spacing and collision issues.

## What It Measures

This metric computes overlap scores that measure:

- **Element collisions**: How much elements physically overlap in the layout
- **Spatial efficiency**: Whether elements are placed with appropriate spacing
- **Layout density**: How tightly elements are packed (which may cause overlaps)

Lower overlap scores generally indicate better layouts with minimal element collisions.

## Metric Details

Implements three overlap metrics from different research works:

1. **LayoutGAN** (Li et al., ICLR 2019): Sum of intersection areas for all element pairs
2. **LayoutGAN++**: Normalized version of LayoutGAN overlap per element
3. **AC-LayoutGAN** (Li et al., TVCG 2021): Ratio-based overlap calculation

Each variant has slightly different calculation methods and sensitivities to overlap patterns.

## Usage

### Installation

```bash
pip install evaluate
```

### Basic Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load("creative-graphic-design/layout-overlap")

# Single layout processing
model_max_length, num_coordinates = 25, 4
bbox = np.random.rand(model_max_length, num_coordinates)
mask = np.random.choice(a=[True, False], size=(model_max_length,))
metric.add(bbox=bbox, mask=mask)
print(metric.compute())
```

### Batch Processing Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load("creative-graphic-design/layout-overlap")

# Batch processing
batch_size, model_max_length, num_coordinates = 512, 25, 4
batch_bbox = np.random.rand(batch_size, model_max_length, num_coordinates)
batch_mask = np.random.choice(a=[True, False], size=(batch_size, model_max_length))
metric.add_batch(bbox=batch_bbox, mask=batch_mask)
print(metric.compute())
```

## Parameters

### Initialization Parameters

This metric does not require any initialization parameters.

### Computation Parameters

- **bbox** (`list` of `lists` of `float`): Bounding boxes in xywh (center-x, center-y, width, height) format
- **mask** (`list` of `lists` of `bool`): Boolean mask indicating valid elements (True) vs padding (False)

**Note**: The mask parameter is essential for handling variable-length layouts where some positions are padding.

## Returns

Returns a dictionary containing three overlap scores:

- **overlap-LayoutGAN** (`array`): Total intersection area across all element pairs (per layout)
- **overlap-LayoutGAN++** (`array`): Normalized overlap per valid element (per layout)
- **overlap-ACLayoutGAN** (`array`): Ratio-based overlap score (per layout)

Each score is an array with one value per layout in the batch.

## Interpretation

### General Interpretation

- **Lower is better** for all three metrics
- **Value of 0**: No overlapping elements (ideal for most layout types)
- **Higher values**: More element overlap, potential layout quality issues

### Metric-Specific Notes

**LayoutGAN**:

- Absolute sum of intersection areas
- Sensitive to both number of overlaps and overlap sizes
- Grows with layout density

**LayoutGAN++**:

- Normalized by number of valid elements
- Better for comparing layouts with different element counts
- Range depends on element sizes and density

**AC-LayoutGAN**:

- Uses area ratios for overlap calculation
- Accounts for element sizes in overlap measurement
- More robust to element size variations

### Use Cases

- **Layout generation evaluation**: Assess whether generated layouts have acceptable spacing
- **Collision detection**: Identify layouts with problematic element overlaps
- **Design quality**: Compare overlap patterns between different generation methods
- **Multi-metric evaluation**: Use alongside alignment and validity metrics

### Key Insights

- **Some overlap may be acceptable** depending on design type (e.g., intentional layering)
- **Context matters**: Poster designs may tolerate more overlap than UI layouts
- **Trade-offs**: Denser layouts naturally have higher overlap potential
- **Compare variants**: Different metrics may highlight different overlap patterns

## Citations

```bibtex
@inproceedings{li2018layoutgan,
  title={LayoutGAN: Generating Graphic Layouts with Wireframe Discriminators},
  author={Li, Jianan and Yang, Jimei and Hertzmann, Aaron and Zhang, Jianming and Xu, Tingfa},
  booktitle={International Conference on Learning Representations},
  year={2019}
}

@article{li2020attribute,
  title={Attribute-conditioned layout gan for automatic graphic design},
  author={Li, Jianan and Yang, Jimei and Zhang, Jianming and Liu, Chang and Wang, Christina and Xu, Tingfa},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  volume={27},
  number={10},
  pages={4039--4048},
  year={2020},
  publisher={IEEE}
}

@inproceedings{kikuchi2021constrained,
  title={Constrained graphic layout generation via latent optimization},
  author={Kikuchi, Kotaro and Simo-Serra, Edgar and Otani, Mayu and Yamaguchi, Kota},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={88--96},
  year={2021}
}
```

## References

- **Paper**: [LayoutGAN (Li et al., ICLR 2019)](https://arxiv.org/abs/1901.06767)
- **Paper**: [Attribute-Conditioned Layout GAN (Li et al., TVCG 2021)](https://arxiv.org/abs/2009.05284)
- **Paper**: [Constrained Graphic Layout Generation (Kikuchi et al., ACM MM 2021)](https://arxiv.org/abs/2108.00871)
- **Reference Implementation (CGL)**: [const_layout metric](https://github.com/ktrk115/const_layout/blob/master/metric.py#L138-L164)
- **Reference Implementation (layout-dm)**: [layout-dm metric](https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L150-L203)
- **Hugging Face Space**: [creative-graphic-design/layout-overlap](https://huggingface.co/spaces/creative-graphic-design/layout-overlap)

## Related Metrics

- [Layout Average IoU](../layout_average_iou/): Measures overlap using IoU metric
- [Layout Overlay](../layout_overlay/): Specialized overlap metric excluding underlay elements
- [Layout Alignment](../layout_alignment/): Evaluates element alignment patterns
- [Layout Validity](../layout_validity/): Checks basic layout validity constraints
