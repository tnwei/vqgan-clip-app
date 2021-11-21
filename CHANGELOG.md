# Changelog

## 1.0 - Nov 21, 2021

Since starting this repo in Jul 2021 as a personal project, I believe the codebase is now sufficiently feature-complete and stable to call it a 1.0 release. 

### What's new

+ Added support for CLIP guided diffusion (https://github.com/tnwei/vqgan-clip-app/pull/8)
+ Improvements to gallery viewer: added pagination (https://github.com/tnwei/vqgan-clip-app/pull/3), improved webpage responsiveness (https://github.com/tnwei/vqgan-clip-app/issues/9)
+ Minor tweaks to VQGAN-CLIP app, added options to control MSE regularization (https://github.com/tnwei/vqgan-clip-app/pull/4) and TV loss (https://github.com/tnwei/vqgan-clip-app/)
+ Reorganized documentation in README.md and `docs/`

### Transitioning to 1.0 for existing users

Update to 1.0 by running `git pull` from your local copy of this repo. No breaking changes are expected, run results from older versions of the codebase should still show up in the gallery viewer.

However, some new packages are needed to support CLIP guided diffusion. You can follow these steps below instead of setting up the Python environment from scratch:

1. In the repo directory, run `git clone https://github.com/crowsonkb/guided-diffusion`
2. `pip install ./guided-diffusion`
3. `pip install lpips`
4. Download diffusion model checkpoints in `download-diffusion-weights.sh`

## Before Oct 2, 2021

VQGAN-CLIP app and basic gallery viewer implemented.