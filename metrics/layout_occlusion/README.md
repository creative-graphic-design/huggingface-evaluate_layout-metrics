---
title: Layout Occlusion
emoji: 🌍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Occlusion

## Description

The Layout Occlusion metric evaluates how much layout elements occlude or cover important visual regions in the background canvas. This metric is particularly important for content-aware layout generation where background imagery should remain visible and not be blocked by poorly placed elements.

## What It Measures

This metric computes the average saliency (visual importance) of canvas regions covered by layout elements:

- **Visual importance coverage**: How much salient (visually important) content is blocked by elements
- **Element placement quality**: Whether elements are placed on less important background regions
- **Content-awareness**: How well the layout respects the underlying visual content

Lower occlusion scores indicate better placement where elements avoid covering important background content.

## Metric Details

- Uses saliency maps to identify visually important regions in the canvas
- Computes average saliency values in areas covered by elements
- Combines two saliency maps for robust evaluation
- From PosterLayout (Hsu et al., CVPR 2023) for content-aware poster design
- Lower scores mean elements are placed on less salient (less important) regions

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
    "creative-graphic-design/layout-occlusion",
    canvas_width=360,
    canvas_height=504
)

# Prepare data
predictions = np.random.rand(1, 25, 4)
gold_labels = np.random.randint(0, 4, size=(1, 25))
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
    "creative-graphic-design/layout-occlusion",
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

- **predictions** (`list` of `lists` of `float`): Normalized bounding boxes in ltrb format
- **gold_labels** (`list` of `lists` of `int`): Class labels for each element (0 = padding)
- **saliency_maps_1** (`list` of `str`): File paths to first set of saliency map images
- **saliency_maps_2** (`list` of `str`): File paths to second set of saliency map images

**Note**: Saliency maps should be grayscale images (0-255) where brighter regions indicate more visually important areas. They will be automatically resized to match canvas dimensions if needed.

## Returns

Returns a `float` value representing the average saliency of occluded regions (range: 0.0 to 1.0).

## Interpretation

- **Lower is better** (range: 0.0 to 1.0)
- **Value ~0.0**: Elements placed on unimportant background regions (ideal)
- **Value ~0.5**: Elements partially cover moderately important regions
- **Value ~1.0**: Elements heavily occlude highly salient background content (problematic)

### Use Cases

- **Content-aware layout generation**: Evaluate if generated layouts respect background imagery
- **Poster/flyer design**: Ensure text and graphics don't block important visual elements
- **Advertisement layout**: Place call-to-action elements without covering key visuals
- **Magazine/presentation layouts**: Balance element placement with background content

### Key Insights

- **Good layouts** minimize occlusion of salient background regions
- **Background-aware models** should achieve lower occlusion scores
- **Trade-off**: Sometimes covering salient regions is necessary for design needs
- **Use with other metrics**: Combine with validity and alignment for comprehensive evaluation

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

- [Layout Utility](../layout_utility/): Measures how well suitable space is utilized
- [Layout Unreadability](../layout_unreadability/): Evaluates text placement on non-flat regions
- [Layout Validity](../layout_validity/): Checks basic validity constraints
