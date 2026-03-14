from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.base_policy import BaseLowdimPolicy


class BeyondMimicTransformerLowdimPolicy(BaseLowdimPolicy):
    """BeyondMimic-style policy with state-latent diffusion + multi-teacher distillation.

    Implemented components:
    1) Transformer diffusion model with causal attention (configured in TransformerForDiffusion).
    2) State-latent conditioning: observations are encoded into compact latent tokens.
    3) Multi-policy distillation: optional teacher action ensemble with learned routing.
    """

    def __init__(
        self,
        model: TransformerForDiffusion,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        latent_dim: int = 64,
        teacher_count: int = 0,
        num_inference_steps=None,
        pred_action_steps_only=False,
        distill_loss_weight: float = 0.2,
        recon_loss_weight: float = 0.1,
        # scheduler.step kwargs
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.latent_dim = latent_dim
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, action_dim),
        )

        self.teacher_count = teacher_count
        self.teacher_router = None
        if teacher_count > 0:
            self.teacher_router = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, teacher_count),
            )

        self.distill_loss_weight = distill_loss_weight
        self.recon_loss_weight = recon_loss_weight

        self.mask_generator = LowdimMaskGenerator(
            action_dim=latent_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )

    def _encode_state_cond(self, obs: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(obs[:, : self.n_obs_steps])

    def conditional_sample(self, condition_data, condition_mask, cond=None, generator=None, **kwargs):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(trajectory, t, cond)
            trajectory = self.noise_scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B = nobs.shape[0]

        cond = self._encode_state_cond(nobs)
        latent_shape = (B, self.horizon, self.latent_dim)
        if self.pred_action_steps_only:
            latent_shape = (B, self.n_action_steps, self.latent_dim)

        cond_data = torch.zeros(size=latent_shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        latent_pred = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            **self.kwargs,
        )

        naction_pred = self.action_decoder(latent_pred)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        return {"action": action, "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        model_groups = self.model.get_optim_groups(weight_decay=weight_decay)
        extra_groups = [
            {"params": self.state_encoder.parameters(), "weight_decay": weight_decay},
            {"params": self.action_encoder.parameters(), "weight_decay": weight_decay},
            {"params": self.action_decoder.parameters(), "weight_decay": weight_decay},
        ]
        if self.teacher_router is not None:
            extra_groups.append(
                {"params": self.teacher_router.parameters(), "weight_decay": weight_decay}
            )
        return torch.optim.AdamW(
            model_groups + extra_groups,
            lr=learning_rate,
            betas=tuple(betas),
        )

    def _compute_distillation_loss(self, state_latent: torch.Tensor, student_action: torch.Tensor, batch: Dict[str, torch.Tensor]):
        if self.teacher_router is None:
            return torch.tensor(0.0, device=student_action.device, dtype=student_action.dtype)
        if "teacher_actions" not in batch:
            return torch.tensor(0.0, device=student_action.device, dtype=student_action.dtype)

        teacher_actions = batch["teacher_actions"]
        # expected shape: (B, T, K, Da)
        if teacher_actions.ndim != 4 or teacher_actions.shape[2] != self.teacher_count:
            return torch.tensor(0.0, device=student_action.device, dtype=student_action.dtype)

        router_logits = self.teacher_router(state_latent.mean(dim=1))
        router_weight = torch.softmax(router_logits, dim=-1)
        mixed_teacher = (teacher_actions * router_weight[:, None, :, None]).sum(dim=2)
        return F.mse_loss(student_action, mixed_teacher)

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize({"obs": batch["obs"], "action": batch["action"]})
        obs = nbatch["obs"]
        action = nbatch["action"]

        state_latent = self._encode_state_cond(obs)

        action_latent = self.action_encoder(action)
        if self.pred_action_steps_only:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            trajectory = action_latent[:, start:end]
        else:
            trajectory = action_latent

        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool) if self.pred_action_steps_only else self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, state_latent)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_mask = ~condition_mask
        diffusion_loss = F.mse_loss(pred, target, reduction="none")
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, "b ... -> b (...)", "mean").mean()

        recon_action = self.action_decoder(action_latent)
        recon_loss = F.mse_loss(recon_action, action)

        distill_loss = self._compute_distillation_loss(
            state_latent=state_latent,
            student_action=recon_action,
            batch=batch,
        )

        return diffusion_loss + self.recon_loss_weight * recon_loss + self.distill_loss_weight * distill_loss
