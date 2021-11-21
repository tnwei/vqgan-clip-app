# Implementation details

## VQGAN-CLIP

Code for the VQGAN-CLIP app mostly came from the z+quantize_method notebook hosted in [EleutherAI/vqgan-clip](https://github.com/EleutherAI/vqgan-clip/tree/main/notebooks). The logic is mostly left unchanged, just refactored to work well with a GUI frontend. 

Prompt weighting was implemented after seeing it in one of the Colab variants floating around the internet. Browsing other notebooks led to adding loss functions like MSE regularization and TV loss for improved image quality.

## CLIP guided diffusion

Code for the CLIP guided diffusion app mostly came from the HQ 512x512 Unconditional notebook, which also can be found in EleutherAI/vqgan-clip. Models from other guided diffusion notebooks are also implemented save for the non-HQ 256x256 version, which did not generate satisfactory results during testing. 