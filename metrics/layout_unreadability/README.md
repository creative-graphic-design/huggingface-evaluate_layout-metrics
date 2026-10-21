---
title: Layout Unreadability
emoji: 🌍
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Unreadability

## Description

The Layout Unreadability metric evaluates whether text elements are placed on visually complex or non-flat background regions that could impair readability. This metric computes the non-flatness (gradient intensity) of regions where text is positioned, helping assess whether text placement respects readability principles in content-aware layout design.

## What It Measures

This metric computes:

- **Background complexity under text**: Gradient intensity in regions occupied by text elements
- **Text readability risk**: Whether text is placed on busy or complex backgrounds
- **Content-awareness**: How well the layout avoids placing text on unsuitable regions

Lower scores indicate better text placement on flat, readable backgrounds.

## Metric Details

- Converts grayscale pixels to float32 before applying signed x- and y-direction Sobel filters
- Computes `sqrt((gx² + gy²) / 2)`, scales each image by its maximum gradient to the 0–255 range, quantizes the scaled gradient map to 8-bit integers, and averages those values under text elements
- Removes underlay/decoration regions from the text mask, then averages scores across all samples
- From PosterLayout (Hsu et al., CVPR 2023) and CGL-GAN methodology
- Lower gradient scores mean text is on flatter, more readable backgrounds

## Usage

### Installation

```bash
pip install evaluate opencv-python
```

### Basic Example

```python
import evaluate
import numpy as np

# Load the metric with canvas dimensions
metric = evaluate.load(
    "creative-graphic-design/layout-unreadability",
    canvas_width=360,
    canvas_height=504,
    text_label_index=1,
    decoration_label_index=3
)

# Prepare data
predictions = np.random.rand(1, 25, 4)  # normalized ltrb coordinates
gold_labels = np.random.randint(0, 4, size=(1, 25))  # class labels
# Paths to canvas background images
image_canvases = ["path/to/canvas_image.jpg"]

score = metric.compute(
    predictions=predictions,
    gold_labels=gold_labels,
    image_canvases=image_canvases
)
print(score)
```

### Batch Processing Example

```python
import evaluate

# Load the metric
metric = evaluate.load(
    "creative-graphic-design/layout-unreadability",
    canvas_width=360,
    canvas_height=504,
    text_label_index=1,
    decoration_label_index=3
)

# Batch processing
batch_size = 128
predictions = np.random.rand(batch_size, 25, 4)
gold_labels = np.random.randint(0, 4, size=(batch_size, 25))
image_canvases = [f"path/to/canvas_{i}.jpg" for i in range(batch_size)]

score = metric.compute(
    predictions=predictions,
    gold_labels=gold_labels,
    image_canvases=image_canvases
)
print(score)
```

## Parameters

### Initialization Parameters

- **canvas_width** (`int`, required): Width of the canvas in pixels
- **canvas_height** (`int`, required): Height of the canvas in pixels
- **text_label_index** (`int`, optional, default=1): Class index for text elements
- **decoration_label_index** (`int`, optional, default=3): Class index for underlay/decoration elements to mask out

### Computation Parameters

- **predictions** (`list` of `lists` of `float`): Normalized bounding boxes in ltrb format (0.0 to 1.0)
- **gold_labels** (`list` of `lists` of `int`): Class labels for each element (0 = padding)
- **image_canvases** (`list` of `str`): File paths to canvas background images

**Note**:

- Canvas images should show the background content (photos, graphics) where layout will be placed
- Underlay/decoration elements are masked out before computing gradients
- Only text elements (text_label_index) are evaluated for readability

## Returns

Returns a `float` value representing the average gradient intensity under text elements (range: 0.0 to 1.0).

## Interpretation

- **Lower is better** (range: 0.0 to 1.0)
- **Value ~0.0**: Text placed on flat, uniform backgrounds (ideal for readability)
- **Value 0.0-0.2**: Good text placement on relatively flat regions
- **Value 0.2-0.4**: Moderate background complexity, may affect readability
- **Value 0.4-0.6**: High background complexity, readability concerns
- **Value > 0.6**: Very complex backgrounds under text (poor placement)

### Use Cases

- **Content-aware poster generation**: Ensure text is readable on background imagery
- **Advertisement layout**: Place call-to-action text on suitable backgrounds
- **Presentation slides**: Validate text visibility on photo backgrounds
- **Magazine/flyer design**: Assess text-background contrast and readability

### Key Insights

- **Readability principle**: Text should be on flat or low-detail backgrounds
- **Design solutions**: Use underlay/decoration elements to create readable regions
- **Trade-off**: Sometimes text must go on complex backgrounds (consider semi-transparent overlays)
- **Context matters**: Title text may tolerate more complexity than body text

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
- **Related**: CGL-GAN text readability evaluation

## Related Metrics

- [Layout Occlusion](../layout_occlusion/): Evaluates coverage of salient regions
- [Layout Utility](../layout_utility/): Measures utilization of suitable space
- [Layout Underlay Effectiveness](../layout_underlay_effectiveness/): Evaluates underlay placement

## Definition and Comparability

The metric computes Sobel gradients in float32 so positive and negative responses are squared without clipping or 8-bit overflow. It then scales the gradient map to 0–255, quantizes it to 8-bit integers, and averages the quantized values under text elements. For the documented fixture, this quantization gives a score of 0.02552; averaging the unquantized scaled values would give 0.02690. This corrects the unreadability calculation in PosterLayout's original code; values are not comparable with numbers reported in papers that used that code.

PosterLayout fixes its canvas at 513 × 750; this metric continues to accept configurable canvas dimensions.
