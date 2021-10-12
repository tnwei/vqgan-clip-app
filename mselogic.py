from typing import Optional, List, Tuple
from PIL import Image
import argparse
import clip
from vqgan_utils import (
    load_vqgan_model,
    MakeCutouts,
    parse_prompt,
    resize_image,
    Prompt,
    synth,
    checkin,
)
import torch
from torchvision.transforms import functional as TF
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
from mse_vqgan_utils import synth_mse, MSEMakeCutouts, noise_gen
import kornia.augmentation as K
from logic import VQGANCLIPRun


class MSEVQGANCLIPRun(VQGANCLIPRun):
    def __init__(
        # Inputs
        self,
        text_input: str = "the first day of the waters",
        vqgan_ckpt: str = "vqgan_imagenet_f16_16384",
        num_steps: int = 300,
        image_x: int = 300,
        image_y: int = 300,
        init_image: Optional[Image.Image] = None,
        image_prompts: List[Image.Image] = [],
        continue_prev_run: bool = False,
        seed: Optional[int] = None,
        # MSE VQGAN-CLIP options from notebook
        use_augs: bool = True,
        noise_fac: float = 0.1,
        use_noise: Optional[float] = None,
        mse_withzeros=True,
        # mse_decay_rate=50,
        # mse_epoches=5,
        # Added options
        mse_weight=0.5,
        mse_weight_decay=0.1,
        mse_weight_decay_steps=50,
    ) -> None:
        super().__init__()
        self.text_input = text_input
        self.vqgan_ckpt = vqgan_ckpt
        self.num_steps = num_steps
        self.image_x = image_x
        self.image_y = image_y
        self.init_image = init_image
        self.image_prompts = image_prompts
        self.continue_prev_run = continue_prev_run
        self.seed = seed

        # Setup ------------------------------------------------------------------------------
        # Split text by "|" symbol
        texts = [phrase.strip() for phrase in text_input.split("|")]
        if texts == [""]:
            texts = []

        # Leaving most of this untouched
        self.args = argparse.Namespace(
            prompts=texts,
            image_prompts=image_prompts,
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            size=[int(image_x), int(image_y)],
            init_image=init_image,
            init_weight=mse_weight,
            clip_model="ViT-B/32",
            vqgan_config=f"assets/{vqgan_ckpt}.yaml",
            vqgan_checkpoint=f"assets/{vqgan_ckpt}.ckpt",
            step_size=0.05,
            cutn=64,
            cut_pow=1.0,
            display_freq=50,
            seed=seed,
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        print("Using device:", device)

        # TODO: MSE regularized options here
        self.iterate_counter = 0
        self.use_augs = use_augs
        self.noise_fac = noise_fac
        self.use_noise = use_noise
        self.mse_withzeros = mse_withzeros
        self.mse_weight_decay = mse_weight_decay
        self.mse_weight_decay_steps = mse_weight_decay_steps

        self.mse_weight = self.args.init_weight
        self.init_mse_weight = self.mse_weight

        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomAffine(
                degrees=30, translate=0.1, p=0.8, padding_mode="border"
            ),  # padding_mode=2
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        )

    def model_init(self, init_image: Image.Image = None) -> None:
        cut_size = self.perceptor.visual.input_resolution

        if self.args.vqgan_checkpoint == "vqgan_openimages_f16_8192.ckpt":
            self.e_dim = 256
            self.n_toks = self.model.quantize.n_embed
            self.z_min = self.model.quantize.embed.weight.min(dim=0).values[
                None, :, None, None
            ]
            self.z_max = self.model.quantize.embed.weight.max(dim=0).values[
                None, :, None, None
            ]
        else:
            self.e_dim = self.model.quantize.e_dim
            self.n_toks = self.model.quantize.n_e
            self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[
                None, :, None, None
            ]
            self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[
                None, :, None, None
            ]

        f = 2 ** (self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MSEMakeCutouts(
            cut_size,
            self.args.cutn,
            cut_pow=self.args.cut_pow,
            augs=self.augs if self.use_augs is True else None,
        )
        n_toks = self.model.quantize.n_e
        toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
        sideX, sideY = toksX * f, toksY * f  # notebook used 16 instead of f here

        if self.seed is not None:
            torch.manual_seed(self.seed)
        else:
            self.seed = torch.seed()  # Trigger a seed, retrieve the utilized seed

        # Initialization order: continue_prev_im, init_image, then only random init
        if init_image is not None:
            init_image = init_image.resize((sideX, sideY), Image.LANCZOS)
            init_image = TF.to_tensor(init_image)

            if self.args.use_noise:
                init_image = init_image + self.args.use_noise * torch.randn_like(
                    init_image
                )

            self.z, *_ = self.model.encode(
                TF.to_tensor(init_image).to(self.device).unsqueeze(0) * 2 - 1
            )
        elif self.args.init_image:
            pil_image = self.args.init_image
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_image = TF.to_tensor(pil_image)

            if self.args.use_noise:
                pil_image = pil_image + self.args.use_noise * torch.randn_like(
                    pil_image
                )

            self.z, *_ = self.model.encode(
                TF.to_tensor(pil_image).to(self.device).unsqueeze(0) * 2 - 1
            )
        else:
            one_hot = F.one_hot(
                torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks
            ).float()

            if self.args.vqgan_checkpoint == "vqgan_openimages_f16_8192.ckpt":
                self.z = one_hot @ self.model.quantize.embed.weight
            else:
                self.z = one_hot @ self.model.quantize.embedding.weight

            self.z = self.z.view([-1, toksY, toksX, self.e_dim]).permute(0, 3, 1, 2)

        if self.mse_withzeros and not self.args.init_image:
            self.z_orig = torch.zeros_like(self.z)
        else:
            self.z_orig = self.z.clone()

        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=self.args.step_size)

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        self.pMs = []

        for prompt in self.args.prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.perceptor.encode_text(
                clip.tokenize(txt).to(self.device)
            ).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for uploaded_image in self.args.image_prompts:
            # path, weight, stop = parse_prompt(prompt)
            # img = resize_image(Image.open(fetch(path)).convert("RGB"), (sideX, sideY))
            img = resize_image(uploaded_image.convert("RGB"), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for seed, weight in zip(
            self.args.noise_prompt_seeds, self.args.noise_prompt_weights
        ):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(
                generator=gen
            )
            self.pMs.append(Prompt(embed, weight).to(self.device))

    def _ascend_txt(self) -> List:
        out = synth_mse(
            self.model,
            self.z,
            quantize=True,
            is_openimages_f16_8192=True
            if self.args.vqgan_checkpoint == "vqgan_openimages_f16_8192.ckpt"
            else False,
        )

        cutouts = self.make_cutouts(out)
        iii = self.perceptor.encode_image(self.normalize(cutouts)).float()

        result = []

        if self.args.init_weight:
            result.append(F.mse_loss(self.z, self.z_orig) * self.mse_weight / 2)

            with torch.no_grad():
                # if not the first step
                # and is time for step change
                # and both weight decay steps and magnitude are nonzero
                # and MSE isn't zero already
                if (
                    self.iterate_counter > 0
                    and self.iterate_counter % self.mse_weight_decay_steps == 0
                    and self.mse_weight_decay != 0
                    and self.mse_weight_decay_steps != 0
                    and self.mse_weight != 0
                ):
                    self.mse_weight = self.mse_weight - self.mse_weight_decay

                    # Don't allow changing sign
                    # Basically, caps MSE at zero if decreasing from positive
                    # But, also prevents MSE from becoming positive if -MSE intended
                    if self.init_mse_weight > 0:
                        self.mse_weight = max(self.mse_weight, 0)
                    else:
                        self.mse_weight = min(self.mse_weight, 0)

                    print(f"updated mse weight: {self.mse_weight}")

        for prompt in self.pMs:
            result.append(prompt(iii))

        return result

    def iterate(self) -> Tuple[List[float], Image.Image]:
        # Forward prop
        self.opt.zero_grad()
        losses = self._ascend_txt()

        # Grab an image
        im: Image.Image = checkin(self.model, self.z)

        # Backprop
        loss = sum(losses)
        loss.backward()
        self.opt.step()
        # with torch.no_grad():
        #     self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

        # Advance iteration counter
        self.iterate_counter += 1

        # Output stuff useful for humans
        return [loss.item() for loss in losses], im
