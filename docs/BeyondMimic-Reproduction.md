# BeyondMimic Reproduction (Guided Diffusion Baseline)

This repository now includes a **BeyondMimic-inspired guided diffusion policy** built on top of the original DiffuseLoco training stack.

## What is implemented

- `GuidedDiffusionTransformerLowdimPolicy`
  - keeps the original diffusion denoising objective;
  - adds an auxiliary guidance head trained with locomotion proxy signals;
  - applies gradient guidance during the reverse diffusion process at inference.
- A dedicated training config:
  - `diffusion_policy/config_files/beyondmimic_guided_diffusion.yaml`

## Key design choices

Because this repository is quadruped-focused and does not ship BeyondMimic humanoid assets/datasets, the reproduction is implemented as a **transferable algorithmic layer** rather than an exact humanoid benchmark clone.

Guidance proxy targets used in training:

1. **Tracking / smoothness**: penalize action discontinuities.
2. **Stability**: penalize large torso-related observation magnitudes (first 3 obs channels).
3. **Energy**: penalize large action magnitudes.

At inference, the policy optimizes a weighted sum of these guidance predictions via trajectory gradients.

## Train

```bash
source env.sh
python scripts/train.py --config-name=beyondmimic_guided_diffusion
```

## Evaluate

```bash
source env.sh
python scripts/eval.py \
  --checkpoint=<PATH_TO_CKPT> \
  --task=cyber2_walk
```

## Suggested next steps for a full BeyondMimic reproduction

- Replace proxy targets with paper-consistent reward/value guidance terms.
- Add humanoid embodiment, retargeting, and motion-tracking task APIs.
- Extend observations/actions and objective terms for upper-body and contact-aware control.
- Add paper-level evaluation metrics and reporting scripts.
