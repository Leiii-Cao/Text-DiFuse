import numpy as np
import torch

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. 
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance_org(
        self, diffusion_stage1,diffusion_stage2, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance_org(self._wrap_diffusion_stage1(diffusion_stage1),self._wrap_diffusion_stage2(diffusion_stage2), *args, **kwargs)
    def p_mean_variance(
        self, diffusion_stage1,diffusion_stage2,Fusion_Control_Model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_diffusion_stage1(diffusion_stage1),self._wrap_diffusion_stage2(diffusion_stage2),Fusion_Control_Model, *args, **kwargs)

    def train_FCM_loss(
        self, diffusion_stage1,diffusion_stage2,Fusion_Control_Model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().train_FCM_loss(self._wrap_diffusion_stage1(diffusion_stage1),self._wrap_diffusion_stage2(diffusion_stage2),Fusion_Control_Model, *args, **kwargs)
        
    def modulated_loss(
        self, diffusion_stage1,diffusion_stage2,Fusion_Control_Model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().modulated_loss(self._wrap_diffusion_stage1(diffusion_stage1),self._wrap_diffusion_stage2(diffusion_stage2),Fusion_Control_Model, *args, **kwargs)
        

    def training_losses(
        self, diffusion_stage1,diffusion_stage2, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_diffusion_stage1(diffusion_stage1),self._wrap_diffusion_stage2(diffusion_stage2), *args, **kwargs)

    def _wrap_diffusion_stage1(self, model):
        if isinstance(model, _WrappedModel_diffusion_stage1):
            return model
        return _WrappedModel_diffusion_stage1(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )
    def _wrap_diffusion_stage2(self, model):
        if isinstance(model, _WrappedModel_diffusion_stage2):
            return model
        return _WrappedModel_diffusion_stage2(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )
    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel_diffusion_stage1:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
        
class _WrappedModel_diffusion_stage2:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts,feature_hidden,feature_skip_list, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts,feature_hidden,feature_skip_list, **kwargs)
