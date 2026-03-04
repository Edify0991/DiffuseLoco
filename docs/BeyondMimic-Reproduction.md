# BeyondMimic Reproduction (Transformer + Distillation + State-Latent)

This update adds a **closer BeyondMimic-style implementation** on top of DiffuseLoco with the three core ideas you asked for:

1. **Transformer diffusion with causal attention**
2. **Multi-policy distillation**
3. **State-latent diffusion modeling**

## Implemented modules

- Main policy:
  - `diffusion_policy/diffusion_policy/policy/beyondmimic_transformer_lowdim_policy.py`
- Training config:
  - `diffusion_policy/config_files/beyondmimic_guided_diffusion.yaml`
- Optional dataset support for teacher ensembles:
  - `diffusion_policy/diffusion_policy/dataset/cyber_dataset.py`

## 1) Transformer diffusion + causal attention

The policy uses `TransformerForDiffusion` as denoiser over latent action trajectories.

In config, `causal_attn: true` enables causality in the diffusion transformer.

## 2) Multi-policy distillation

A teacher router network learns mixture weights over multiple teacher policies.

- Input: encoded state latent
- Output: teacher mixture weights via softmax
- Distillation objective: MSE between student action reconstruction and weighted teacher action mixture

Expected teacher data shape in a batch:

- `teacher_actions`: `(B, T, K, Da)` where `K` is teacher count

Dataset path support is provided through `teacher_action_key` in `CyberDogDataset`.

## 3) State-latent diffusion

Instead of directly diffusing actions in raw action space:

- observations are encoded into state latents (`state_encoder`), used as diffusion condition;
- actions are encoded into latent trajectories (`action_encoder`), which become diffusion targets;
- decoded actions are produced by `action_decoder`.

Training objective combines:

- diffusion denoising loss in latent space;
- action reconstruction loss;
- multi-policy distillation loss (if teacher actions are available).

## Run training

```bash
source env.sh
python scripts/train.py --config-name=beyondmimic_guided_diffusion
```

## Notes

- This repository is still quadruped-oriented; full humanoid paper parity also needs humanoid assets, task APIs, and benchmark scripts.
- The algorithmic structure requested in your feedback is now implemented in code and configurable for further extension.
