import gradio as gr
import argparse
import math
from pathlib import Path
import sys

sys.path.append("./taming-transformers")

from PIL import Image
import requests
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from CLIP import clip

from utils import (
    load_vqgan_model,
    MakeCutouts,
    parse_prompt,
    resize_image,
    Prompt,
    fetch,
    synth,
    checkin,
)

# Hacky method to preserve state
class State:
    model = None
    perceptor = None
    prev_im = None


state = State()


def run(
    text_input: str = "the first day of the waters",
    vqgan_ckpt: str = "vqgan_imagenet_f16_16384",
    num_steps: int = 300,
    image_x=300,
    image_y=300,
    continue_prev_run: bool = False,
):
    # Leaving most of this untouched
    args = argparse.Namespace(
        prompts=[text_input],
        image_prompts=[],
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size=[int(image_y), int(image_x)],
        # init_image=None,
        init_weight=0.0,
        # clip.available_models()
        # ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']
        # Visual Transformer seems to be the smallest
        clip_model="ViT-B/32",
        vqgan_config=f"assets/{vqgan_ckpt}.yaml",
        vqgan_checkpoint=f"assets/{vqgan_ckpt}.ckpt",
        step_size=0.05,
        cutn=64,
        cut_pow=1.0,
        display_freq=50,
        seed=None,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load models
    if continue_prev_run is True:
        model = state.model
        perceptor = state.perceptor
    else:
        model = state.model = load_vqgan_model(
            args.vqgan_config, args.vqgan_checkpoint
        ).to(device)
        perceptor = state.perceptor = (
            clip.load(args.clip_model, jit=False)[0]
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    # Initialize models
    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2 ** (model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # if args.init_image:
    if continue_prev_run:
        # pil_image = Image.open(fetch(args.init_image)).convert('RGB')
        pil_image = state.prev_im
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(
            torch.randint(n_toks, [toksY * toksX], device=device), n_toks
        ).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    ## Initialize prompts
    pMs = []

    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(fetch(path)).convert("RGB"), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    def ascend_txt():
        out = synth(model, z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

        result = []

        if args.init_weight:
            result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

        for prompt in pMs:
            result.append(prompt(iii))

        return result

    for i in range(num_steps):
        opt.zero_grad()
        lossAll = ascend_txt()
        im = checkin(i, lossAll, model, z)
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))

    state.prev_im = im
    return im


with open("README.md", "r") as f:
    long_description = f.read()


iface = gr.Interface(
    fn=run,
    inputs=[
        "text",
        gr.inputs.Radio(
            [
                "vqgan_imagenet_f16_1024",
                "vqgan_imagenet_f16_16384",
                "coco",
                "faceshq",
                "sflickr",
                "wikiart_16384",
                "wikiart_1024",
            ]
        ),
        gr.inputs.Slider(minimum=20, maximum=2000, step=20, default=300),
        gr.inputs.Number(default=300, label="Xdim"),
        gr.inputs.Number(default=300, label="Ydim"),
        gr.inputs.Checkbox(label="Continue previous run"),
    ],
    outputs=["image"],
    title="VQGAN-CLIP",
    live=False,  # Live reloads based on input which is NOT what we want
    server_port=7860,
    theme="default",  # default, dark, huggingface
    allow_screenshot=True,
    allow_flagging=False,
    description="Simple Gradio webapp for local experimentation of VQGAN-CLIP",
    article=long_description,
)

iface.launch(share=False)
