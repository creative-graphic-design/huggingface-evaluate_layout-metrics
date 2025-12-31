---
title: Layout Generative Model Scores
emoji: 📊
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Layout Generative Model Scores

## Description

The Layout Generative Model Scores metric computes a comprehensive set of distribution-based metrics to evaluate generative models for layout generation. This metric compares feature distributions between real and generated layouts using state-of-the-art evaluation methods.

## What It Measures

This metric implements several widely-used generative model evaluation scores:

1. **FID (Fréchet Inception Distance)**: Measures the distance between real and generated feature distributions
2. **Precision**: Measures the proportion of generated samples that are realistic
3. **Recall**: Measures the proportion of real samples covered by the generated distribution
4. **Density**: Estimates the density of the generated distribution
5. **Coverage**: Measures the diversity of generated samples

These metrics provide a holistic view of both the quality (precision, FID) and diversity (recall, coverage) of generated layouts.

## Metric Details

- **FID**: Computed using Fréchet distance between Gaussian distributions fitted to real and fake features
- **PRDC (Precision, Recall, Density, Coverage)**: Computed using k-nearest neighbors in feature space
- Requires feature vectors extracted from layouts (typically using a pre-trained neural network)
- All metrics are distribution-based, providing robust evaluation across large sample sets

## Usage

### Installation

```bash
pip install evaluate prdc pytorch-fid
```

### Basic Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load("creative-graphic-design/layout-generative-model-scores")

# Single processing
feat_size = 256
feats_real = np.random.rand(feat_size)
feats_fake = np.random.rand(feat_size)
metric.add(feats_real=feats_real, feats_fake=feats_fake)
print(metric.compute())
```

### Batch Processing Example

```python
import evaluate
import numpy as np

# Load the metric
metric = evaluate.load("creative-graphic-design/layout-generative-model-scores")

# Batch processing (recommended for meaningful statistics)
batch_size, feat_size = 512, 256
feats_real = np.random.rand(batch_size, feat_size)
feats_fake = np.random.rand(batch_size, feat_size)
metric.add_batch(feats_real=feats_real, feats_fake=feats_fake)
print(metric.compute())
```

## Parameters

### Initialization Parameters

- **nearest_k** (`int`, optional, default=5): Number of nearest neighbors to use for PRDC computation

### Computation Parameters

- **feats_real** (`list` of `list` of `float`): Feature vectors extracted from real layouts (shape: N × feature_dim)
- **feats_fake** (`list` of `list` of `float`): Feature vectors extracted from generated/fake layouts (shape: N × feature_dim)

**Note**: Features are typically extracted using a pre-trained neural network (e.g., layout encoder) that converts layouts into fixed-dimensional feature vectors.

## Returns

Returns a dictionary containing:

- **fid** (`float`): Fréchet Inception Distance
- **precision** (`float`): Precision score (0.0 to 1.0)
- **recall** (`float`): Recall score (0.0 to 1.0)
- **density** (`float`): Density score
- **coverage** (`float`): Coverage score (0.0 to 1.0)

## Interpretation

### FID (Fréchet Inception Distance)

- **Lower is better**: Measures similarity between real and generated distributions
- **Value of 0**: Perfect match between distributions
- **Typical range**: 0 to ∞ (in practice, usually 0-300 for layout tasks)
- **Usage**: Primary metric for assessing overall generative quality

### Precision

- **Higher is better** (range: 0.0 to 1.0)
- Measures what proportion of generated samples are realistic
- High precision = most generated layouts look realistic
- Low precision = many unrealistic generated layouts

### Recall

- **Higher is better** (range: 0.0 to 1.0)
- Measures what proportion of real distribution is covered by generated samples
- High recall = generated layouts cover the diversity of real layouts
- Low recall = generated layouts miss parts of the real distribution (mode collapse)

### Density

- Estimates how densely generated samples are packed in feature space
- Higher values indicate more samples in covered regions

### Coverage

- **Higher is better** (range: 0.0 to 1.0)
- Measures the proportion of real samples within the generated distribution's support
- Similar to recall but computed differently
- High coverage = good diversity in generation

### Trade-offs

- **Precision vs Recall**: Common trade-off in generative models
  - High precision, low recall: Safe but limited generation (mode dropping)
  - Low precision, high recall: Diverse but potentially unrealistic generation
  - Goal: Balance both for quality and diversity

## Citations

```bibtex
@article{heusel2017gans,
  title={Gans trained by a two time-scale update rule converge to a local nash equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@inproceedings{naeem2020reliable,
  title={Reliable fidelity and diversity metrics for generative models},
  author={Naeem, Muhammad Ferjad and Oh, Seong Joon and Uh, Youngjung and Choi, Yunjey and Yoo, Jaejun},
  booktitle={International Conference on Machine Learning},
  pages={7176--7185},
  year={2020},
  organization={PMLR}
}
```

## References

- **Paper**: [GANs Trained by a Two Time-Scale Update Rule (Heusel et al., NeurIPS 2017)](https://arxiv.org/abs/1706.08500)
- **Paper**: [Reliable Fidelity and Diversity Metrics for Generative Models (Naeem et al., ICML 2020)](https://arxiv.org/abs/2002.09797)
- **Reference Implementation (FID + PRDC)**: [layout-dm metric implementation](https://github.com/CyberAgentAILab/layout-dm/blob/main/src/trainer/trainer/helpers/metric.py#L37-L59)
- **PRDC Library**: [generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc)
- **PyTorch FID**: [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- **Hugging Face Space**: [creative-graphic-design/layout-generative-model-scores](https://huggingface.co/spaces/creative-graphic-design/layout-generative-model-scores)

## Related Metrics

- [Layout Average IoU](../layout_average_iou/): Measures element overlap
- [Layout Maximum IoU](../layout_maximum_iou/): Compares layout similarity
- [Layout Validity](../layout_validity/): Checks layout validity constraints
