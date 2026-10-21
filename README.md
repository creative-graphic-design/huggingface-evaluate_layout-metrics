# 🤗 Layout Evaluation Metrics by Huggingface Evaluate

[![CI](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/ci.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/ci.yaml)
[![Deploy spaces](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/deploy.yaml/badge.svg)](https://github.com/creative-graphic-design/huggingface-evaluate_layout-metrics/actions/workflows/deploy.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A collection of metrics to evaluate layout generation that can be easily used in 🤗 huggingface [evaluate](https://huggingface.co/docs/evaluate/index).

## Generative Model Evaluation Metrics

| 📊 Metric | 🤗 Space                                                                                                                                         | 📝 Paper                                   |
| :-------- | :----------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------- |
| FID       | [`creative-graphic-design/layout-generative-model-scores`](https://huggingface.co/spaces/creative-graphic-design/layout-generative-model-scores) | Heusel+ NeurIPS'17[^1], Naeem+ ICML'20[^4] |

## Basic Layout Quality Metrics

| 📊 Metric | 🤗 Space                                                                                                                 | 📝 Paper                                                 |
| :-------- | :----------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------- |
| Max. IoU  | [`creative-graphic-design/layout-maximum-iou`](https://huggingface.co/spaces/creative-graphic-design/layout-maximum-iou) | Kikuchi+ ACMMM'21[^6]                                    |
| LTSim     | [`creative-graphic-design/layout-ltsim`](https://huggingface.co/spaces/creative-graphic-design/layout-ltsim)             | Otani+ arXiv'24[^10]                                     |
| Avg. IoU  | [`creative-graphic-design/layout-average-iou`](https://huggingface.co/spaces/creative-graphic-design/layout-average-iou) | Arroyo+ CVPR'21[^7], Kong+ ECCV'22[^8]                   |
| Alignment | [`creative-graphic-design/layout-alignment`](https://huggingface.co/spaces/creative-graphic-design/layout-alignment)     | Lee+ ECCV'20[^3], Li+ TVCG'21[^5], Kikuchi+ ACMMM'21[^6] |
| Overlap   | [`creative-graphic-design/layout-overlap`](https://huggingface.co/spaces/creative-graphic-design/layout-overlap)         | Li+ ICLR'19[^2], Li+ TVCG'21[^5], Kikuchi+ ACMMM'21[^6]  |

## Content-Aware Layout Metrics (PosterLayout)

| 📊 Metric              | 🤗 Space                                                                                                                                       | 📝 Paper                          |
| :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------- |
| Validity               | [`creative-graphic-design/layout-validity`](https://huggingface.co/spaces/creative-graphic-design/layout-validity)                             | Hsu+ CVPR'23[^9]                  |
| Occlusion              | [`creative-graphic-design/layout-occlusion`](https://huggingface.co/spaces/creative-graphic-design/layout-occlusion)                           | Hsu+ CVPR'23[^9]                  |
| Overlay                | [`creative-graphic-design/layout-overlay`](https://huggingface.co/spaces/creative-graphic-design/layout-overlay)                               | Hsu+ CVPR'23[^9]                  |
| Underlay Effectiveness | [`creative-graphic-design/layout-underlay-effectiveness`](https://huggingface.co/spaces/creative-graphic-design/layout-underlay-effectiveness) | Hsu+ CVPR'23[^9]                  |
| Unreadability          | [`creative-graphic-design/layout-unreadability`](https://huggingface.co/spaces/creative-graphic-design/layout-unreadability)                   | Hsu+ CVPR'23[^9]                  |
| Utility                | [`creative-graphic-design/layout-utility`](https://huggingface.co/spaces/creative-graphic-design/layout-utility)                               | Hsu+ CVPR'23[^9]                  |
| Non-Alignment          | [`creative-graphic-design/layout-non-alignment`](https://huggingface.co/spaces/creative-graphic-design/layout-non-alignment)                   | Hsu+ CVPR'23[^9], Li+ TVCG'21[^5] |

# Usage

- Install [`evaluate`](https://huggingface.co/docs/evaluate/index) library

```shell
pip install evaluate
```

- Load the layout metric and then compute the score

```python
import evaluate
import numpy as np

# Load the evaluation metric named "creative-graphic-design/layout-alignment"
alignment_score = evaluate.load("creative-graphic-design/layout-alignment")

# `batch_bbox` is a tensor representing (batch_size, max_num_elements, coordinates)
# and `batch_mask` is a boolean tensor representing (batch_size, max_num_elements).
batch_bbox = np.random.rand(512, 25, 4)
# Note that padded fields will be set to `False`
batch_mask = np.full((512, 25), fill_value=True)

# Add the batch of bboxes and masks to the metric
alignment_score.add_batch(batch_bbox=batch_bbox, batch_mask=batch_mask)
# Perform the computation of the evaluation metric
alignment_score.compute()
```

## Vendor references

Official references are vendored as read-only submodules when no installable release exists; otherwise parity uses a pinned test-only dependency. Published metrics do not import vendored trees. `layout_generative_model_scores` depends on `prdc` and `pytorch-fid` at runtime; parity tests check PRDC precision, recall, density, and coverage against an independent implementation of the Naeem et al. definitions and FID against an independent implementation of the Fréchet distance formula.

- [`CyberAgentAILab/layout-dm`](https://github.com/CyberAgentAILab/layout-dm), Apache-2.0, tag `v1.0.0` at commit `93630ba8f5b04103f3fa8506732a7702cc837463`, used only by parity tests.
- [`mayu-ot/ltsim`](https://github.com/mayu-ot/ltsim), MIT, tag `v1.0.0-alpha` at commit `544b8bccfce4630591589cdec5c5473392190039`, used only by parity tests.
- [`ktrk115/const_layout`](https://github.com/ktrk115/const_layout), AGPL-3.0, `master` at commit `5287480505939345543fff0b9f2e5d541e6f84e2`, vendored at `vendor/const-layout` as a read-only secondary cross-check for alignment, overlap, and maximum IoU in parity tests only; never imported by a published metric.
- [`PKU-ICST-MIPL/PosterLayout-CVPR2023`](https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023), no license file provided in the upstream repository, `main` at commit `44ee576471f3c06d06fc632f13e3f96d0c381847`, vendored at `vendor/pku-poster-layout` as a read-only tests-only reference with owner approval; no source is copied and no published metric imports it.

## Contributing and deployment

Add a metric under `metrics/<metric_name>` with its README, hyphenated module, `app.py`, and `pyproject.toml`; the deployment workflow discovers the directory automatically, so a new metric needs no workflow copy.

On a pull request, `validate-space (<metric>)` runs only for changed metric directories or shared deployment inputs such as `uv.lock`, `pyproject.toml`, `scripts/deploy_space.py`, and `deploy.yaml`; it builds the four-file Space payload and checks its local metadata and exported requirements without Hugging Face credentials.

After CI succeeds on `main`, `deploy (<metric>)` runs for the same changed set, checks whether the Gradio Space exists, creates it when possible, and uploads the payload with a commit message containing the source commit.

Existing Spaces are checked before creation so an organization plan restriction does not block their uploads; if a Space is missing and creation is blocked, the owner must upgrade the organization plan, choose paid hardware for that Space, or decide not to publish it under the organization.

The main-branch change detector compares `HEAD^..HEAD`; this assumes pull requests are squash-merged, as configured in the repository. A multi-commit direct push would include only its last commit and can miss earlier changes.

Space names are derived by replacing underscores with hyphens, so `metrics/layout_average_iou` deploys to `creative-graphic-design/layout-average-iou`.

The deploy job uses only the repository secret `HF_TOKEN`, passed to `huggingface_hub` over HTTP; `HF_USERNAME` and `HF_EMAIL` are not needed. The namespace is fixed in the workflow.

The owner must configure the Hugging Face token as follows: token type: fine-grained; organization: creative-graphic-design; check exactly these three under "Repositories": "Read contents of repos in selected orgs", "Create repos in selected orgs and read/write/manage repos created by this token", "Write contents/settings of repos in selected orgs"; leave every other box (Inference, Org settings, Billing, Collections, Resource Groups, Jobs, Audit log, Service accounts, access requests, discussions/PRs) unchecked.

"Read contents of repos in selected orgs" and "Write contents/settings of repos in selected orgs" are required for `upload_folder` to the 12 existing Spaces.

"Create repos in selected orgs and read/write/manage repos created by this token" is required for `create_repo(repo_type="space", exist_ok=True)` of new Spaces.

Store the token as the repository secret `HF_TOKEN`. To rotate it, create and approve a replacement, update that secret, run the Deploy spaces workflow from the Actions tab with Redeploy all metrics checked, and revoke the old token.

The `Deploy spaces` badge near the top covers the single deployment workflow for every metric.

To roll back a deployment, revert the source commit on `main`; CI then redeploys the reverted metric payload.

## Reference

[^1]: Heusel, Martin, et al. "[Gans trained by a two time-scale update rule converge to a local nash equilibrium.](https://arxiv.org/abs/1706.08500)" Advances in neural information processing systems 30 (2017).
[^2]: Li, Jianan, et al. "[LayoutGAN: Generating Graphic Layouts with Wireframe Discriminators.](https://arxiv.org/abs/1901.06767)" International Conference on Learning Representations. 2019.
[^3]: Lee, Hsin-Ying, et al. "[Neural design network: Graphic layout generation with constraints.](https://arxiv.org/abs/1912.09421)" Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16. Springer International Publishing, 2020.
[^4]: Naeem, Muhammad Ferjad, et al. "[Reliable fidelity and diversity metrics for generative models.](https://arxiv.org/abs/2002.09797)" International Conference on Machine Learning. PMLR, 2020.
[^5]: Li, Jianan, et al. "[Attribute-conditioned layout gan for automatic graphic design.](https://arxiv.org/abs/2009.05284)" IEEE Transactions on Visualization and Computer Graphics 27.10 (2020): 4039-4048.
[^6]: Kikuchi, Kotaro, et al. "[Constrained graphic layout generation via latent optimization.](https://arxiv.org/abs/2108.00871)" Proceedings of the 29th ACM International Conference on Multimedia. 2021.
[^7]: Arroyo, Diego Martin, Janis Postels, and Federico Tombari. "[Variational transformer networks for layout generation.](https://arxiv.org/abs/2104.02416)" Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
[^8]: Kong, Xiang, et al. "[BLT: bidirectional layout transformer for controllable layout generation.](https://arxiv.org/abs/2112.05112)" European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
[^9]: Hsu, Hsiao Yuan, et al. "[Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout.](https://arxiv.org/abs/2303.15937)" Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
[^10]: Otani, Mayu, et al. "[LTSim: Layout Transportation-based Similarity Measure for Evaluating Layout Generation.](https://arxiv.org/abs/2407.12356)" arXiv preprint arXiv:2407.12356. 2024.
