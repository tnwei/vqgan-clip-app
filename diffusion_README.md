Based on https://github.com/EleutherAI/vqgan-clip/tree/main/notebooks#clip_guided_diffusionipynb-on-colab. This code lives on https://github.com/tnwei/vqgan-clip-app/tree/guided-diffusion, plan to have it merged to main when ready.

In addition to the VQGAN-CLIP app's installation instructions, requires running the following:

1. `curl -L -o assets/256x256_diffusion_uncond.pt -C - 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'`
2. `git clone https://github.com/openai/guided-diffusion`
3. `pip install ./guided-diffusion`