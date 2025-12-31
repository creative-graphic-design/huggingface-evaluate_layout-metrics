---
title: Layout Utility
emoji: 🏆
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Utility

## Description

The Layout Utility metric evaluates how efficiently a layout utilizes suitable space for element placement. It measures the utilization rate of regions that are appropriate for arranging elements, as determined by the negative (inverted) saliency map. This metric helps assess whether layouts make effective use of available non-salient (less visually important) space.

## What It Measures

This metric computes:

- **Space utilization**: How much of the suitable (low-saliency) space is occupied by elements
- **Placement efficiency**: Whether elements are placed in appropriate regions
- **Canvas coverage**: How well the layout fills available design space

Higher scores indicate better utilization of suitable space, though extremely high scores might indicate overcrowding.

## Metric Details

- Uses the **negative/inverted saliency map** (S') to identify suitable placement regions
- Suitable space = regions with LOW visual importance (complement of salient areas)
- Computes the ratio of element-covered suitable space to total suitable space
- From PosterLayout (Hsu et al., CVPR 2023) for content-aware poster design
- Higher utility means elements effectively use non-salient space

## Usage

### Installation

```bash
pip install evaluate
```

### Basic Example

```python
import evaluate
import numpy as np

# Load the metric with canvas dimensions
metric = evaluate.load(
    "creative-graphic-design/layout-utility",
    canvas_width=360,
    canvas_height=504
)

# Prepare data
predictions = np.random.rand(1, 25, 4)  # normalized ltrb coordinates
gold_labels = np.random.randint(0, 4, size=(1, 25))  # class labels
# Paths to saliency map images (grayscale, 0-255)
saliency_maps_1 = ["path/to/saliency_map_1.png"]
saliency_maps_2 = ["path/to/saliency_map_2.png"]

score = metric.compute(
    predictions=predictions,
    gold_labels=gold_labels,
    saliency_maps_1=saliency_maps_1,
    saliency_maps_2=saliency_maps_2
)
print(score)
```

### Batch Processing Example

```python
import evaluate

# Load the metric
metric = evaluate.load(
    "creative-graphic-design/layout-utility",
    canvas_width=360,
    canvas_height=504
)

# Batch processing
batch_size = 128
predictions = np.random.rand(batch_size, 25, 4)
gold_labels = np.random.randint(0, 4, size=(batch_size, 25))
saliency_maps_1 = [f"path/to/saliency_{i}_1.png" for i in range(batch_size)]
saliency_maps_2 = [f"path/to/saliency_{i}_2.png" for i in range(batch_size)]

score = metric.compute(
    predictions=predictions,
    gold_labels=gold_labels,
    saliency_maps_1=saliency_maps_1,
    saliency_maps_2=saliency_maps_2
)
print(score)
```

## Parameters

### Initialization Parameters

- **canvas_width** (`int`, required): Width of the canvas in pixels
- **canvas_height** (`int`, required): Height of the canvas in pixels

### Computation Parameters

- **predictions** (`list` of `lists` of `float`): Normalized bounding boxes in ltrb format (0.0 to 1.0)
- **gold_labels** (`list` of `lists` of `int`): Class labels for each element (0 = padding)
- **saliency_maps_1** (`list` of `str`): File paths to first set of saliency map images
- **saliency_maps_2** (`list` of `str`): File paths to second set of saliency map images

**Note**:

- Saliency maps should be grayscale images (0-255) where brighter = more salient/important
- The metric internally inverts these to find suitable (non-salient) placement regions
- Very small elements (< 0.1% of canvas) are filtered out

## Returns

Returns a `float` value representing the utilization rate of suitable space (range: 0.0 to 1.0).

## Interpretation

- **Higher is generally better** (range: 0.0 to 1.0)
- **Value ~1.0**: Very high utilization of suitable space (may indicate dense layout)
- **Value 0.6-0.9**: Good space utilization
- **Value 0.3-0.6**: Moderate utilization, room for more elements
- **Value 0.0-0.3**: Low utilization, underused suitable space
- **Value ~0.0**: Very sparse layout, most suitable space unused

### Use Cases

- **Content-aware layout generation**: Evaluate if layouts efficiently use available space
- **Poster/flyer design**: Assess whether designs make good use of non-salient regions
- **Space efficiency analysis**: Compare layout density across different designs
- **Design quality**: Identify layouts that are too sparse or potentially too dense

### Key Insights

- **Balance is important**: Neither too sparse nor too dense is ideal
- **Context-dependent**: Optimal utility varies by design type (minimalist vs. information-dense)
- **Suitable space concept**: Based on inverted saliency - where elements CAN go, not where they SHOULD avoid
- **Combine with occlusion**: High utility + low occlusion = good space utilization without blocking important content
- **Not absolute quality**: High utility alone doesn't guarantee good design

### Interpretation Considerations

- Very high utility (>0.9) might indicate overcrowding
- Very low utility (<0.2) might indicate wasteful use of space
- Optimal range typically 0.4-0.8 depending on design intent

## Citations

```bibtex
@inproceedings{hsu2023posterlayout,
  title={Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout},
  author={Hsu, Hsiao Yuan and He, Xiangteng and Peng, Yuxin and Kong, Hao and Zhang, Qing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6018--6026},
  year={2023}
}
```

## References

- **Paper**: [PosterLayout (Hsu et al., CVPR 2023)](https://arxiv.org/abs/2303.15937)
- **Reference Implementation**: [PosterLayout eval.py](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/eval.py#L144-L171)

## Related Metrics

- [Layout Occlusion](../layout_occlusion/): Evaluates coverage of salient (important) regions
- [Layout Validity](../layout_validity/): Checks basic validity constraints
- [Layout Unreadability](../layout_unreadability/): Evaluates text placement quality
