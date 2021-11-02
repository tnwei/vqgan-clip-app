import clip
import sys
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from kornia import augmentation, filters
from torch import nn
from torch.nn import functional as F
import math
import lpips

sys.path.append("./guided-diffusion")

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def parse_prompt(prompt):
    vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


class CLIPGuidedDiffusion256:
    """
    Deprecated, the HQ versions require code from a forked and modified
    guided-diffusion repo, which no longer works with this.
    """

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
                "diffusion_steps": 1000,  # not supposed to change this?
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


class CLIPGuidedDiffusion256HQ:
    def __init__(
        self,
        prompt: str,
        batch_size: int = 1,
        clip_guidance_scale: float = 1000,
        seed: int = 0,
        num_steps: int = 1000,
        continue_prev_run: bool = True,
    ) -> None:
        # Default config
        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": str(
                    num_steps
                ),  # modify this to decrease timesteps
                "image_size": 256,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_checkpoint": False,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        self.prompts = [prompt]  # TODO: Prompt splitting
        self.image_prompts = []  # TODO
        self.batch_size = batch_size
        # Controls how much the image should look like the prompt.
        self.clip_guidance_scale = clip_guidance_scale

        # Controls the smoothness of the final output.
        self.tv_scale = 150  # TODO add control widget

        # Controls how far out of range RGB values are allowed to be.
        self.range_scale = 50  # TODO add control widget

        self.cutn = 16  # TODO add control widget

        # Removed, repeat batches by triggering a new run
        # self.n_batches = 1

        self.init_image = None  # TODO add control widget

        # This enhances the effect of the init image, a good value is 1000.
        self.init_scale = 1000  # TODO add control widget

        # This needs to be between approx. 200 and 500 when using an init image.
        # Higher values make the output look more like the init.
        self.skip_timesteps = 0  # TODO add control widget

        self.seed = seed
        self.continue_prev_run = continue_prev_run

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

    def load_model(
        self,
        model_file_loc="assets/256x256_diffusion_uncond.pt",
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
            if self.init_image is None:
                self.lpips_model = None
            else:
                self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

            return self.model, self.diffusion, self.clip_model

    def cond_fn(self, x, t, out, y=None):
        n = x.shape[0]
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[self.cur_t]
        x_in = out["pred_xstart"] * fac + x * (1 - fac)
        clip_in = self.normalize(self.make_cutouts(x_in.add(1).div(2)))
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = spherical_dist_loss(
            image_embeds.unsqueeze(1), self.target_embeds.unsqueeze(0)
        )
        dists = dists.view([self.cutn, n, -1])
        losses = dists.mul(self.weights).sum(2).mean(0)
        tv_losses = tv_loss(x_in)
        range_losses = range_loss(out["pred_xstart"])
        loss = (
            losses.sum() * self.clip_guidance_scale
            + tv_losses.sum() * self.tv_scale
            + range_losses.sum() * self.range_scale
        )
        # TODO: Implement init image
        # if init is not None and init_scale:
        #     init_losses = lpips_model(x_in, init)
        #     loss = loss + init_losses.sum() * init_scale
        return -torch.autograd.grad(loss, x)[0]

    def model_init(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.make_cutouts = MakeCutouts(self.clip_size, self.cutn)
        self.side_x = self.side_y = self.model_config["image_size"]

        self.target_embeds, self.weights = [], []

        for prompt in self.prompts:
            txt, weight = parse_prompt(prompt)
            self.target_embeds.append(
                self.clip_model.encode_text(clip.tokenize(txt).to(self.device)).float()
            )
            self.weights.append(weight)

        # TODO: Implement image prompt parsing
        # for prompt in self.image_prompts:
        #     path, weight = parse_prompt(prompt)
        #     img = Image.open(fetch(path)).convert('RGB')
        #     img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        #     batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        #     embed = clip_model.encode_image(normalize(batch)).float()
        #     target_embeds.append(embed)
        #     weights.extend([weight / cutn] * cutn)

        self.target_embeds = torch.cat(self.target_embeds)
        self.weights = torch.tensor(self.weights, device=self.device)
        if self.weights.sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")
        self.weights /= self.weights.sum().abs()

        self.init = None
        # if init_image is not None:
        #     init = Image.open(fetch(init_image)).convert('RGB')
        #     init = init.resize((side_x, side_y), Image.LANCZOS)
        #     init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if self.model_config["timestep_respacing"].startswith("ddim"):
            sample_fn = self.diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = self.diffusion.p_sample_loop_progressive

        self.cur_t = self.diffusion.num_timesteps - self.skip_timesteps - 1

        self.samples = sample_fn(
            self.model,
            (self.batch_size, 3, self.side_y, self.side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=self.cond_fn,
            progress=True,
            skip_timesteps=self.skip_timesteps,
            init_image=self.init,
            randomize_class=True,
            cond_fn_with_grad=True,
        )

        self.samplesgen = enumerate(self.samples)

    def iterate(self):
        self.cur_t -= 1
        _, sample = next(self.samplesgen)

        ims = []
        for _, image in enumerate(sample["pred_xstart"]):
            im = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
            ims.append(im)

        return ims


class CLIPGuidedDiffusion512HQ:
    def __init__(
        self,
        prompt: str,
        batch_size: int = 1,
        clip_guidance_scale: float = 1000,
        seed: int = 0,
        num_steps: int = 1000,
        continue_prev_run: bool = True,
    ) -> None:
        # Default config
        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": True,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": str(
                    num_steps
                ),  # modify this to decrease timesteps
                "image_size": 512,
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

        self.prompts = [prompt]  # TODO: Prompt splitting
        self.image_prompts = []  # TODO
        self.batch_size = batch_size
        # Controls how much the image should look like the prompt.
        self.clip_guidance_scale = clip_guidance_scale

        # Controls the smoothness of the final output.
        self.tv_scale = 150  # TODO add control widget

        # Controls how far out of range RGB values are allowed to be.
        self.range_scale = 50  # TODO add control widget

        self.cutn = 40  # TODO add control widget
        self.cut_pow = 0.5  # TODO add control widget

        # Removed, repeat batches by triggering a new run
        # self.n_batches = 1

        self.init_image = None  # TODO add control widget

        # This enhances the effect of the init image, a good value is 1000.
        self.init_scale = 1000  # TODO add control widget

        # This needs to be between approx. 200 and 500 when using an init image.
        # Higher values make the output look more like the init.
        self.skip_timesteps = 0  # TODO add control widget

        self.seed = seed
        self.continue_prev_run = continue_prev_run

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

    def load_model(
        self,
        model_file_loc="assets/256x256_diffusion_uncond.pt",
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

            for name, param in self.model.named_parameters():
                if "qkv" in name or "norm" in name or "proj" in name:
                    param.requires_grad_()

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
            # In 256 and 512 uncond, but not in 512 conditional
            # if self.init_image is None:
            #     self.lpips_model = None
            # else:
            #     self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

            return self.model, self.diffusion, self.clip_model

    def cond_fn(self, x, t, y=None):
        # 512 HQ notebook different from 256 HQ notebook
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=self.device, dtype=torch.long) * self.cur_t
            out = self.diffusion.p_mean_variance(
                self.model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
            )
            fac = self.diffusion.sqrt_one_minus_alphas_cumprod[self.cur_t]
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            clip_in = self.normalize(self.make_cutouts(x_in.add(1).div(2)))
            image_embeds = (
                self.clip_model.encode_image(clip_in).float().view([self.cutn, n, -1])
            )
            dists = spherical_dist_loss(image_embeds, self.target_embeds.unsqueeze(0))
            losses = dists.mean(0)
            tv_losses = tv_loss(x_in)
            loss = (
                losses.sum() * self.clip_guidance_scale
                + tv_losses.sum() * self.tv_scale
            )
            # TODO: Implement init image
            return -torch.autograd.grad(loss, x)[0]

    def model_init(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.make_cutouts = MakeCutouts(self.clip_size, self.cutn, self.cut_pow)
        self.side_x = self.side_y = self.model_config["image_size"]

        self.target_embeds, self.weights = [], []

        # Prompt weightage parsing in 256 HQ, but not in 512 HQ
        for prompt in self.prompts:
            txt, weight = parse_prompt(prompt)
            self.target_embeds.append(
                self.clip_model.encode_text(clip.tokenize(txt).to(self.device)).float()
            )
            self.weights.append(weight)

        # TODO: Implement image prompt parsing
        # for prompt in self.image_prompts:
        #     path, weight = parse_prompt(prompt)
        #     img = Image.open(fetch(path)).convert('RGB')
        #     img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        #     batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        #     embed = clip_model.encode_image(normalize(batch)).float()
        #     target_embeds.append(embed)
        #     weights.extend([weight / cutn] * cutn)

        self.target_embeds = torch.cat(self.target_embeds)
        self.weights = torch.tensor(self.weights, device=self.device)
        if self.weights.sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")
        self.weights /= self.weights.sum().abs()

        self.init = None
        # if init_image is not None:
        #     init = Image.open(fetch(init_image)).convert('RGB')
        #     init = init.resize((side_x, side_y), Image.LANCZOS)
        #     init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if self.model_config["timestep_respacing"].startswith("ddim"):
            sample_fn = self.diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = self.diffusion.p_sample_loop_progressive

        self.cur_t = self.diffusion.num_timesteps - self.skip_timesteps - 1

        self.samples = sample_fn(
            self.model,
            (self.batch_size, 3, self.side_y, self.side_x),
            clip_denoised=False,
            model_kwargs={
                "y": torch.zeros(
                    [self.batch_size], device=self.device, dtype=torch.long
                )
            },
            cond_fn=self.cond_fn,
            progress=True,
            skip_timesteps=self.skip_timesteps,
            init_image=self.init,
            randomize_class=True,
            # cond_fn_with_grad=True, # not in 512 HQ
        )

        self.samplesgen = enumerate(self.samples)

    def iterate(self):
        self.cur_t -= 1
        _, sample = next(self.samplesgen)

        ims = []
        for _, image in enumerate(sample["pred_xstart"]):
            im = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
            ims.append(im)

        return ims


class CLIPGuidedDiffusion512HQUncond:
    def __init__(
        self,
        prompt: str,
        batch_size: int = 1,
        clip_guidance_scale: float = 1000,
        seed: int = 0,
        num_steps: int = 1000,
        continue_prev_run: bool = True,
    ) -> None:
        # Default config
        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": str(
                    num_steps
                ),  # modify this to decrease timesteps
                "image_size": 512,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_checkpoint": False,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        self.prompts = [prompt]  # TODO: Prompt splitting
        self.image_prompts = []  # TODO
        self.batch_size = batch_size
        # Controls how much the image should look like the prompt.
        self.clip_guidance_scale = clip_guidance_scale

        # Controls the smoothness of the final output.
        self.tv_scale = 150  # TODO add control widget

        # Controls how far out of range RGB values are allowed to be.
        self.range_scale = 50  # TODO add control widget

        self.cutn = 32  # TODO add control widget
        self.cutn_batches = 2  # TODO add control widget
        self.cut_pow = 0.5  # TODO add control widget

        # Removed, repeat batches by triggering a new run
        # self.n_batches = 1

        self.init_image = None  # TODO add control widget

        # This enhances the effect of the init image, a good value is 1000.
        self.init_scale = 1000  # TODO add control widget

        # This needs to be between approx. 200 and 500 when using an init image.
        # Higher values make the output look more like the init.
        self.skip_timesteps = 0  # TODO add control widget

        self.seed = seed
        self.continue_prev_run = continue_prev_run

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

    def load_model(
        self,
        model_file_loc="assets/256x256_diffusion_uncond.pt",
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

            # for name, param in self.model.named_parameters():
            #     if 'qkv' in name or 'norm' in name or 'proj' in name:
            #         param.requires_grad_()

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

            if self.init_image is None:
                self.lpips_model = None
            else:
                self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

            return self.model, self.diffusion, self.clip_model

    def cond_fn(self, x, t, out, y=None):
        n = x.shape[0]
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[self.cur_t]
        x_in = out["pred_xstart"] * fac + x * (1 - fac)
        x_in_grad = torch.zeros_like(x_in)
        for i in range(self.cutn_batches):
            clip_in = self.normalize(self.make_cutouts(x_in.add(1).div(2)))
            image_embeds = self.clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(
                image_embeds.unsqueeze(1), self.target_embeds.unsqueeze(0)
            )
            dists = dists.view([self.cutn, n, -1])
            losses = dists.mul(self.weights).sum(2).mean(0)
            x_in_grad += (
                torch.autograd.grad(losses.sum() * self.clip_guidance_scale, x_in)[0]
                / self.cutn_batches
            )
        tv_losses = tv_loss(x_in)
        range_losses = range_loss(out["pred_xstart"])
        loss = tv_losses.sum() * self.tv_scale + range_losses.sum() * self.range_scale
        if self.init is not None and self.init_scale:
            init_losses = self.lpips_model(x_in, self.init)
            loss = loss + init_losses.sum() * self.init_scale
        x_in_grad += torch.autograd.grad(loss, x_in)[0]
        grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
        return grad

    def model_init(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.make_cutouts = MakeCutouts(self.clip_size, self.cutn, self.cut_pow)
        self.side_x = self.side_y = self.model_config["image_size"]

        self.target_embeds, self.weights = [], []

        for prompt in self.prompts:
            txt, weight = parse_prompt(prompt)
            self.target_embeds.append(
                self.clip_model.encode_text(clip.tokenize(txt).to(self.device)).float()
            )
            self.weights.append(weight)

        # TODO: Implement image prompt parsing
        # for prompt in self.image_prompts:
        #     path, weight = parse_prompt(prompt)
        #     img = Image.open(fetch(path)).convert('RGB')
        #     img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        #     batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        #     embed = clip_model.encode_image(normalize(batch)).float()
        #     target_embeds.append(embed)
        #     weights.extend([weight / cutn] * cutn)

        self.target_embeds = torch.cat(self.target_embeds)
        self.weights = torch.tensor(self.weights, device=self.device)
        if self.weights.sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")
        self.weights /= self.weights.sum().abs()

        self.init = None
        # if init_image is not None:
        #     init = Image.open(fetch(init_image)).convert('RGB')
        #     init = init.resize((side_x, side_y), Image.LANCZOS)
        #     init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if self.model_config["timestep_respacing"].startswith("ddim"):
            sample_fn = self.diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = self.diffusion.p_sample_loop_progressive

        self.cur_t = self.diffusion.num_timesteps - self.skip_timesteps - 1

        self.samples = sample_fn(
            self.model,
            (self.batch_size, 3, self.side_y, self.side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=self.cond_fn,
            progress=True,
            skip_timesteps=self.skip_timesteps,
            init_image=self.init,
            randomize_class=True,
            cond_fn_with_grad=True,
        )

        self.samplesgen = enumerate(self.samples)

    def iterate(self):
        self.cur_t -= 1
        _, sample = next(self.samplesgen)

        ims = []
        for _, image in enumerate(sample["pred_xstart"]):
            im = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
            ims.append(im)

        return ims
