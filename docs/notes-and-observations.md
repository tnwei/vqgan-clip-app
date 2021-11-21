# Notes and observations

## Generated image size

Generated image size is bound by GPU VRAM available. The reference notebook default to use 480x480. One of the notebooks in the thoughts section below uses 640x512. For reference, an RTX2060 can barely manage 300x300. You can use image upscaling tools such as [Waifu2X](https://github.com/nagadomi/waifu2x) or [Real ESRGAN](https://github.com/xinntao/Real-ESRGAN) to further upscale the generated image beyond VRAM limits. Just be aware that smaller generated images fundamentally contain less complexity than larger images. 

## GPU VRAM consumption

Following are GPU VRAM consumption read from `nvidia-smi`, note that your mileage may vary.

For VQGAN-CLIP, using the `vqgan_imagenet_f16_1024` model checkpoint:

| Resolution| VRAM Consumption |
| ----------| ---------------- |
| 300 x 300 | 4,829 MiB        |
| 480 x 480 | 8,465 MiB        |
| 640 x 360 | 8,169 MiB        |
| 640 x 480 | 10,247 MiB       |
| 800 x 450 | 13,977 MiB       |
| 800 x 600 | 18,157 MiB       |
| 960 x 540 | 15,131 MiB       |
| 960 x 720 | 19,777 MiB       |
| 1024 x 576| 17,175 MiB       |
| 1024 x 768| 22,167 MiB       |
| 1280 x 720| 24,353 MiB       |

For CLIP guided diffusion, using various checkpoints:

| Method            | VRAM Consumption |
| ------------------| ---------------- |
| 256x256 HQ Uncond | 9,199 MiB        |
| 512x512 HQ Cond   | 15,079 MiB       |
| 512x512 HQ Uncond | 15,087 MiB       |

## CUDA out of memory error

If you're getting a CUDA out of memory error on your first run, it is a sign that the image size is too large. If you were able to generate images of a particular size prior, then you might have a CUDA memory leak and need to restart the application. Still figuring out where the leak is from.

## VQGAN weights and art style

In the download links are trained on different datasets. You might find them suitable for generating different art styles. The `sflickr` dataset is skewed towards generating landscape images while the `faceshq` dataset is skewed towards generating faces. If you have no art style preference, the ImageNet weights do remarkably well. In fact, VQGAN-CLIP can be conditioned to generate specific styles, thanks to the breadth of understanding supplied by the CLIP model (see tips section).

Some weights have multiple versions, e.g. ImageNet 1024 and Image 16384. The number represents the codebook (latent space) dimensionality. For more info, refer to the [VQGAN repo](https://github.com/CompVis/taming-transformers), also linked in the intro above.

## How many steps should I let the models run?

A good starting number to try is 1000 steps for VQGAN-CLIP, and 2000 steps for CLIP guided diffusion.

There is no ground rule on how many steps to run to get a good image, images generated are also not guaranteed to be interesting. YMMV as the image generation process depends on how well CLIP is able to understand the given prompt(s).

## Reproducibility

Replicating a run with the same configuration, model checkpoint and seed will allow recreating the same output albeit with very minor variations. There exists a tiny bit of stochasticity that can't be eliminated, due to how the underlying convolution operators are implemented in CUDA. The underlying CUDA operations used by cuDNN can vary depending on differences in hardware or plain noise (see [Pytorch docs on reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)). Using fully deterministic algorithms isn't an option as some CUDA operations do not have deterministic counterparts (e.g. `upsample_bicubic2d_backward_cuda`)

In practice, the variations are not easy to spot unless a side-by-side comparison is done. 