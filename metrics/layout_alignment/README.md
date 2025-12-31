---
title: Layout Alignment
emoji: 📊
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.17.0
app_file: app.py
pinned: false
---

# Layout Alignment

## Description

The Layout Alignment metric evaluates how well layout elements are aligned with each other. This metric implements alignment scoring methods from multiple research papers, providing a comprehensive assessment of spatial organization and visual harmony in graphic layouts.

## What It Measures

This metric computes alignment scores that quantify how elements in a layout adhere to alignment principles:

- **Edge alignment**: How well element edges (left, right, top, bottom, center) align with each other
- **Spatial relationships**: Detection of common alignment patterns (grids, columns, rows)
- **Visual coherence**: Overall harmony created by consistent element positioning

Well-aligned layouts typically score lower (less alignment violation) and appear more professional and organized.

## Metric Details

Implements alignment metrics from multiple influential layout generation papers:

- **NDN-Net (Lee et al., ECCV 2020)**: Neural Design Network alignment evaluation
- **AC-GAN (Li et al., TVCG 2021)**: Attribute-Conditioned GAN alignment metrics
- **CGL (Kikuchi et al., ACM MM 2021)**: Constrained Graphic Layout alignment scores

The metric analyzes element positioning to detect alignment relationships and violations.

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
metric = evaluate.load("creative-graphic-design/layout-alignment")

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
metric = evaluate.load("creative-graphic-design/layout-alignment")

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

- **bbox** (`list` of `lists` of `int`): Bounding boxes for elements in normalized coordinates
- **mask** (`list` of `lists` of `bool`): Boolean mask indicating valid elements (True) vs padding (False)

**Note**: The mask parameter is crucial for handling variable-length layouts, where padding elements should be excluded from computation.

## Returns

Returns a dictionary containing multiple alignment scores from different methods:

- Different variants measuring alignment quality from various perspectives
- Specific score names depend on the implementation details from referenced papers

## Interpretation

- **Lower values generally indicate better alignment** (fewer alignment violations)
- **Value interpretation depends on specific score variant**:
  - Some scores measure alignment violations (lower is better)
  - Others measure alignment quality (higher is better)
- **Typical use**: Compare relative scores between different layout generation methods

### Key Insights

- **Professional layouts** tend to have good alignment scores due to consistent spatial relationships
- **Grid-based layouts** typically achieve better alignment than freeform designs
- **Alignment patterns** (left-aligned, centered, etc.) are important for visual hierarchy

## Citations

```bibtex
@inproceedings{lee2020neural,
  title={Neural design network: Graphic layout generation with constraints},
  author={Lee, Hsin-Ying and Jiang, Lu and Essa, Irfan and Le, Phuong B and Gong, Haifeng and Yang, Ming-Hsuan and Yang, Weilong},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part III 16},
  pages={491--506},
  year={2020},
  organization={Springer}
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

- **Paper**: [Neural Design Network (Lee et al., ECCV 2020)](https://arxiv.org/abs/1912.09421)
- **Paper**: [Attribute-Conditioned Layout GAN (Li et al., TVCG 2021)](https://arxiv.org/abs/2009.05284)
- **Paper**: [Constrained Graphic Layout Generation (Kikuchi et al., ACM MM 2021)](https://arxiv.org/abs/2108.00871)
- **Hugging Face Space**: [creative-graphic-design/layout-alignment](https://huggingface.co/spaces/creative-graphic-design/layout-alignment)

## Related Metrics

- [Layout Non-Alignment](../layout_non_alignment/): Measures spatial non-alignment between elements
- [Layout Overlap](../layout_overlap/): Evaluates element overlap and spacing
- [Layout Validity](../layout_validity/): Checks basic layout validity constraints
