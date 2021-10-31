Based on the diffusion models hosted at https://github.com/EleutherAI/vqgan-clip/tree/main/notebooks. This WIP code lives on https://github.com/tnwei/vqgan-clip-app/tree/guided-diffusion, plan to have it merged to main when ready.

In addition to the VQGAN-CLIP app's installation instructions, requires running the following:

1. After cloning this repo, run `git checkout -b guided-diffusion origin/guided-diffusion` to download this branch. You can now switch between the `main` branch and `guided-diffusion` with `git checkout`.
2. Download the guided diffusion model file:
    + 256x256: `curl -L -o assets/256x256_diffusion_uncond.pt -C - 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'`
    + 512x512 class conditional model: `curl -L -o assets/512x512_diffusion.pt -C - 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt'`
    + 512x512 unconditional model: `curl -L -o assets/512x512_diffusion_uncond_finetune_008100.pt --http1.1 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'`
3. `git clone https://github.com/crowsonkb/guided-diffusion`
4. `pip install ./guided-diffusion`
5. `pip install lpips`

Run the app by calling `streamlit run diffusion_app.py` instead.