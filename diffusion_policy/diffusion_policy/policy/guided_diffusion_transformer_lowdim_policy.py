from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import (
    DiffusionTransformerLowdimPolicy,
)


class GuidedDiffusionTransformerLowdimPolicy(DiffusionTransformerLowdimPolicy):
    """BeyondMimic-style guided diffusion policy.

    This module keeps DiffuseLoco's denoising training objective and adds:
      1) a lightweight guidance value head trained with proxy objectives
      2) test-time gradient guidance over the denoised trajectory

    The implementation is intentionally model-agnostic so it can run on top of
    the existing low-dimensional locomotion datasets in this repository.
    """

    def __init__(
        self,
        model: TransformerForDiffusion,
        noise_scheduler: DDPMScheduler,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_cond=False,
        pred_action_steps_only=False,
        guidance_scale: float = 0.0,
        guidance_lr: float = 0.05,
        guidance_steps: int = 1,
        guidance_loss_weight: float = 0.1,
        guidance_hidden_dim: int = 128,
        objective_weights: Dict[str, float] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            obs_as_cond=obs_as_cond,
            pred_action_steps_only=pred_action_steps_only,
            **kwargs,
        )

        if objective_weights is None:
            objective_weights = {
                "tracking": 1.0,
                "stability": 0.5,
                "energy": 0.1,
            }

        self.guidance_scale = guidance_scale
        self.guidance_lr = guidance_lr
        self.guidance_steps = guidance_steps
        self.guidance_loss_weight = guidance_loss_weight
        self.objective_weights = objective_weights

        cond_feature_dim = obs_dim if obs_as_cond else (obs_dim + action_dim)
        guidance_input_dim = action_dim + cond_feature_dim
        self.guidance_head = nn.Sequential(
            nn.Linear(guidance_input_dim, guidance_hidden_dim),
            nn.SiLU(),
            nn.Linear(guidance_hidden_dim, guidance_hidden_dim),
            nn.SiLU(),
            nn.Linear(guidance_hidden_dim, 3),
        )

    def _compute_guidance_features(
        self, trajectory: torch.Tensor, cond: torch.Tensor = None
    ) -> torch.Tensor:
        traj_feat = trajectory.mean(dim=1)
        if cond is None:
            cond_feat = torch.zeros(
                trajectory.shape[0],
                self.guidance_head[0].in_features - self.action_dim,
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
        else:
            cond_feat = cond.mean(dim=1)
        return torch.cat([traj_feat, cond_feat], dim=-1)

    def _proxy_guidance_targets(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # motion smoothness (higher is better)
        tracking_score = -((action[:, 1:] - action[:, :-1]) ** 2).mean(dim=(1, 2))

        # torso stability proxy using first 3 observation channels
        stable_dims = min(obs.shape[-1], 3)
        stability_score = -(obs[..., :stable_dims].abs()).mean(dim=(1, 2))

        # control-effort regularity
        energy_score = -(action.abs()).mean(dim=(1, 2))

        return torch.stack([tracking_score, stability_score, energy_score], dim=-1)

    def _guidance_objective(self, trajectory: torch.Tensor, cond: torch.Tensor = None):
        guidance_pred = self.guidance_head(
            self._compute_guidance_features(trajectory=trajectory, cond=cond)
        )
        weights = torch.tensor(
            [
                self.objective_weights.get("tracking", 1.0),
                self.objective_weights.get("stability", 0.5),
                self.objective_weights.get("energy", 0.1),
            ],
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        return (guidance_pred * weights).sum(dim=-1)

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        cond=None,
        generator=None,
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, cond)
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample

            if self.guidance_scale > 0.0:
                for _ in range(self.guidance_steps):
                    guided_traj = trajectory.detach().requires_grad_(True)
                    objective = self._guidance_objective(guided_traj, cond=cond).sum()
                    grad = torch.autograd.grad(objective, guided_traj)[0]
                    grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    trajectory = (
                        trajectory
                        + self.guidance_scale * self.guidance_lr * grad / grad_norm
                    )
                    trajectory[condition_mask] = condition_data[condition_mask]

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        model_groups = self.model.get_optim_groups(weight_decay=weight_decay)
        guidance_group = {
            "params": self.guidance_head.parameters(),
            "weight_decay": weight_decay,
        }
        return torch.optim.AdamW(
            model_groups + [guidance_group],
            lr=learning_rate,
            betas=tuple(betas),
        )

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:, : self.n_obs_steps, :]
            if self.pred_action_steps_only:
                start = self.n_obs_steps - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        diffusion_loss = F.mse_loss(pred, target, reduction="none")
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, "b ... -> b (...)", "mean").mean()

        guidance_pred = self.guidance_head(
            self._compute_guidance_features(trajectory=trajectory.detach(), cond=cond)
        )
        guidance_target = self._proxy_guidance_targets(obs=obs, action=action)
        guidance_loss = F.mse_loss(guidance_pred, guidance_target)

        return diffusion_loss + self.guidance_loss_weight * guidance_loss
