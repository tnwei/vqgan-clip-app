import clip
import sys
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from kornia import augmentation, filters
from torch import nn
from torch.nn import functional as F
import math

sys.path.append("./guided-diffusion")

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class CLIPGuidedDiffusion:
    def __init__(
        self,
        prompt: str,
        batch_size: int = 1,
        clip_guidance_scale: float = 2750,
        seed: int = 0,
        num_steps: int = 500,
        continue_prev_run: bool = True,
    ) -> None:
        # Default config
        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": num_steps * 2,
                "rescale_timesteps": False,
                "timestep_respacing": str(num_steps),
                "image_size": 256,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        self.prompt = prompt
        self.batch_size = batch_size
        self.clip_guidance_scale = clip_guidance_scale
        self.seed = seed
        self.continue_prev_run = continue_prev_run

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

    def load_model(
        self,
        model_file_loc="256x256_diffusion_uncond.pt",
        prev_model=None,
        prev_diffusion=None,
        prev_clip_model=None,
    ) -> None:
        if (
            self.continue_prev_run is True
            and prev_model is not None
            and prev_diffusion is not None
            and prev_clip_model is not None
        ):
            self.model = prev_model
            self.diffusion = prev_diffusion
            self.clip_model = prev_clip_model

            self.clip_size = self.clip_model.visual.input_resolution
            self.normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )

        else:
            self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
            self.model.load_state_dict(torch.load(model_file_loc, map_location="cpu"))
            self.model.eval().requires_grad_(False).to(self.device)
            if self.model_config["use_fp16"]:
                self.model.convert_to_fp16()

            self.clip_model = (
                clip.load("ViT-B/16", jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.device)
            )

            self.clip_size = self.clip_model.visual.input_resolution
            self.normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )

            return self.model, self.diffusion, self.clip_model

    def cond_fn(self, x, t, y=None):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_()
            sigma = min(24, self.diffusion.sqrt_recipm1_alphas_cumprod[self.cur_t] / 4)
            print(sigma)
            kernel_size = max(math.ceil((sigma * 6 + 1) / 2) * 2 - 1, 3)
            x_blur = filters.gaussian_blur2d(
                x_in, (kernel_size, kernel_size), (sigma, sigma)
            )
            clip_in = F.interpolate(
                self.aug(x_blur.add(1).div(2)),
                (self.clip_size, self.clip_size),
                mode="bilinear",
                align_corners=False,
            )
            image_embed = self.clip_model.encode_image(self.normalize(clip_in)).float()
            losses = spherical_dist_loss(image_embed, self.text_embed)
            grad = -torch.autograd.grad(losses.sum(), x_in)[0]
            return grad * self.clip_guidance_scale

    def model_init(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.text_embed = self.clip_model.encode_text(
            clip.tokenize(self.prompt).to(self.device)
        ).float()

        translate_by = 8 / self.clip_size
        if translate_by:
            self.aug = augmentation.RandomAffine(
                0, (translate_by, translate_by), padding_mode="border", p=1
            )
        else:
            self.aug = nn.Identity()

        self.cur_t = self.diffusion.num_timesteps - 1

        self.samplesgen = enumerate(
            self.diffusion.p_sample_loop_progressive(
                self.model,
                (
                    self.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=True,
                model_kwargs={},
                cond_fn=self.cond_fn,
                progress=True,
            )
        )

    def iterate(self):
        self.cur_t -= 1
        _, sample = next(self.samplesgen)

        ims = []
        for _, image in enumerate(sample["pred_xstart"]):
            im = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
            ims.append(im)

        return ims
