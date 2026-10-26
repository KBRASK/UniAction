<div align="center">

<h1>UniAction: Unified Action Conditioning for World Models</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2610.32977-b31b1b)](https://arxiv.org/abs/2610.32977)
[![Project Page](https://img.shields.io/badge/Project_Page-Link-green)](https://uniaction.github.io/)
[![🤗 HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange)](https://huggingface.co/BowenXue/UniAction)
[![🤗 Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/BowenXue/UniAction-Train)

</div>

## Abstract

Action-conditioned world models are typically designed for individual action types, fragmenting scarce action-annotated data across separate models. We introduce **UniAction**, a unified action-conditioning framework that represents heterogeneous actions as sparse action elements, each with a discrete semantic identity and continuous spatiotemporal coordinates. These coordinates map directly into the pretrained video model's native positional space, allowing actions and video to interact within a shared spatiotemporal structure. Cardinality-calibrated attention addresses the bias introduced by different numbers of action elements. Across **force, camera control, hand pose, and robot pose**, UniAction improves action following over conditioning alternatives, and a single jointly trained model outperforms action-specific experts.

## Getting Started

### Installation

Use Python 3.11 and CUDA-enabled PyTorch. From the repository root:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu129
pip install -e '.[fast,overlay]'
```

### Inference

Download the [UniAction checkpoints](https://huggingface.co/BowenXue/UniAction) and [Wan2.2-I2V-A14B backbone](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) into `checkpoints/UniAction` and `checkpoints/Wan2.2-I2V-A14B`, respectively. Start with the example:

```python
from uniaction import UniActionPipeline, conditions

with open('examples/1/prompt.txt') as f:
    prompt = f.read().strip()

pipe = UniActionPipeline.from_pretrained(
    'checkpoints/Wan2.2-I2V-A14B',
    'checkpoints/UniAction',
)
pipe.generate(
    'examples/1/first-frame.png',
    prompt,
    conditions.load_condition('examples/1/original.npz'),
    guidance='counterfactual', text_guidance=3, action_guidance=3,
    output='output.mp4',
    overlay=True,
)
```

Change the folder and `.npz` filename to use any of the examples. For your own actions, replace the condition with:

| Action | Condition |
| --- | --- |
| Hand pose | `conditions.hand_pose(uv, visible)` |
| Force | `conditions.force(start, end)` |
| Camera control | `conditions.camera('W')` — supports WASD/IJKL |
| Robot pose | `conditions.robot_pose(uv, visible)` |

Coordinates are normalized `(x, y)` in the input image. Pose trajectories have shape `(81, 12, 2)`; the optional visibility mask has shape `(81, 12)`. Point ordering is defined in [conditions.py](src/uniaction/conditions.py).

### Training

Download the [training set](https://huggingface.co/datasets/BowenXue/UniAction-Train) into `datasets/UniAction-Train/`, then train:

```bash
pip install -e '.[fast,train,data]'

uniaction-train \
  --base-model checkpoints/Wan2.2-I2V-A14B \
  --sources datasets/UniAction-Train/training-sources.json \
  --output runs/uniaction
```

Training samples and preprocesses 81 frames at 832 × 480, encodes them on the fly and keeps the VAE on the GPU. For faster training, prepare the caches first:

```bash
uniaction-cache encode \
  --base-model checkpoints/Wan2.2-I2V-A14B \
  --sources datasets/UniAction-Train/training-sources.json \
  --output caches

uniaction-train \
  --base-model checkpoints/Wan2.2-I2V-A14B \
  --cache-manifest caches/training.json \
  --output runs/uniaction
```

Training uses the visible GPUs and saves intermediate checkpoints automatically. Add `--resume` to continue an interrupted run. The finished model is saved in `runs/uniaction/checkpoint`; load it with `trained=True` in `UniActionPipeline.from_pretrained`.

### Evaluation

Download the [evaluation set](https://huggingface.co/datasets/BowenXue/UniAction-Eval) and run evaluation:

```bash
python3.11 -m venv .venv-eval
source .venv-eval/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[eval]'
uniaction-eval similarity --video output.mp4 --reference reference.mp4 --output metrics.json
```

## Citation

```bibtex
@article{xue2026uniaction,
  title={UniAction: Unified Action Conditioning for World Models},
  author={Xue, Bowen and Li, Zizhang and Yu, Hong-Xing and Wu, Jiajun},
  journal={arXiv preprint arXiv:2610.32977},
  year={2026}
}
```

## Acknowledgments

UniAction builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), with FlashAttention and Quack acceleration. Evaluation uses [CoTracker](https://github.com/facebookresearch/co-tracker) and [ViPE](https://github.com/nv-tlabs/vipe). We thank their authors and contributors.

Code is released under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution. Model weights, datasets and third-party assets retain their respective licenses.
