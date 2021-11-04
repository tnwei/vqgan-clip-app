Based on the diffusion models hosted at https://github.com/EleutherAI/vqgan-clip/tree/main/notebooks. 

In addition to the VQGAN-CLIP app's installation instructions, requires running the following:

1. After cloning this repo, run `git checkout -b guided-diffusion origin/guided-diffusion` to download this branch. You can now switch between the `main` branch and `guided-diffusion` with `git checkout`.
2. Install the additional packages:
    + `git clone https://github.com/crowsonkb/guided-diffusion`
    + `pip install ./guided-diffusion`
    + `pip install lpips`
3. Download the pretrained weights and config files using links in the `download-diffusion-weights.sh` script. Note that that all of the links are commented out by default. Recommend to download one by one, as some of the downloads can take a while.
    
Run the app by calling `streamlit run diffusion_app.py`.