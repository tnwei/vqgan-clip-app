Based on the diffusion models hosted at https://github.com/EleutherAI/vqgan-clip/tree/main/notebooks. This WIP code lives on https://github.com/tnwei/vqgan-clip-app/tree/guided-diffusion, plan to have it merged to main when ready.

In addition to the VQGAN-CLIP app's installation instructions, requires running the following:

1. After cloning this repo, run `git checkout -b guided-diffusion origin/guided-diffusion` to download this branch. You can now switch between the `main` branch and `guided-diffusion` with `git checkout`.
2. Download the pretrained weights and config files using links in the `download-diffusion-weights.sh` script. Note that that all of the links are commented out by default. Recommend to download one by one, as some of the downloads can take a while.
3. Update python environment with `conda env update -f environment.yml` as two new packages are needed. You can also do this manually:
    + `git clone https://github.com/crowsonkb/guided-diffusion`
    + `pip install ./guided-diffusion`
    + `pip install lpips`

Run the app by calling `streamlit run diffusion_app.py` instead.