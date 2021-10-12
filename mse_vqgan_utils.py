import math
import torch
from torch import nn
import torch.nn.functional as F
import io
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
from PIL import Image
from torchvision.transforms import functional as TF
import sys
import kornia.augmentation as K

from vqgan_utils import resample, clamp_with_grad, replace_grad, vector_quantize

sys.path.append("./taming-transformers")


def noise_gen(shape):
    n, c, h, w = shape
    noise = torch.zeros([n, c, 1, 1])
    for i in reversed(range(5)):
        h_cur, w_cur = h // 2 ** i, w // 2 ** i
        noise = F.interpolate(
            noise, (h_cur, w_cur), mode="bicubic", align_corners=False
        )
        noise += torch.randn([n, c, h_cur, w_cur]) / 5
    return noise


def one_sided_clip_loss(input, target, labels=None, logit_scale=100):
    input_normed = F.normalize(input, dim=-1)
    target_normed = F.normalize(target, dim=-1)
    logits = input_normed @ target_normed.T * logit_scale
    if labels is None:
        labels = torch.arange(len(input), device=logits.device)
    return F.cross_entropy(logits, labels)


class MSEMakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0, augs=None, noise_fac=0.1):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = augs
        self.noise_fac = noise_fac

        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []

        min_size_width = min(sideX, sideY)

        for ii in range(self.cutn):

            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)

        if self.augs is not None:
            cutouts = self.augs(cutouts)

        if self.noise_fac:
            facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(
                0, self.noise_fac
            )
            cutouts = cutouts + facs * torch.randn_like(cutouts)

        return clamp_with_grad(cutouts, 0, 1)


class TVLoss(nn.Module):
    def forward(self, input):
        input = F.pad(input, (0, 1, 0, 1), "replicate")
        x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
        y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
        diff = x_diff ** 2 + y_diff ** 2 + 1e-8
        return diff.mean(dim=1).sqrt().mean()


class GaussianBlur2d(nn.Module):
    def __init__(self, sigma, window=0, mode="reflect", value=0):
        super().__init__()
        self.mode = mode
        self.value = value
        if not window:
            window = max(math.ceil((sigma * 6 + 1) / 2) * 2 - 1, 3)
        if sigma:
            kernel = torch.exp(
                -((torch.arange(window) - window // 2) ** 2) / 2 / sigma ** 2
            )
            kernel /= kernel.sum()
        else:
            kernel = torch.ones([1])
        self.register_buffer("kernel", kernel)

    def forward(self, input):
        n, c, h, w = input.shape
        input = input.view([n * c, 1, h, w])
        start_pad = (self.kernel.shape[0] - 1) // 2
        end_pad = self.kernel.shape[0] // 2
        input = F.pad(
            input, (start_pad, end_pad, start_pad, end_pad), self.mode, self.value
        )
        input = F.conv2d(input, self.kernel[None, None, None, :])
        input = F.conv2d(input, self.kernel[None, None, :, None])
        return input.view([n, c, h, w])


class EMATensor(nn.Module):
    """implmeneted by Katherine Crowson"""

    def __init__(self, tensor, decay):
        super().__init__()
        self.tensor = nn.Parameter(tensor)
        self.register_buffer("biased", torch.zeros_like(tensor))
        self.register_buffer("average", torch.zeros_like(tensor))
        self.decay = decay
        self.register_buffer("accum", torch.tensor(1.0))
        self.update()

    @torch.no_grad()
    def update(self):
        if not self.training:
            raise RuntimeError("update() should only be called during training")

        self.accum *= self.decay
        self.biased.mul_(self.decay)
        self.biased.add_((1 - self.decay) * self.tensor)
        self.average.copy_(self.biased)
        self.average.div_(1 - self.accum)

    def forward(self):
        if self.training:
            return self.tensor
        return self.average


def synth_mse(model, z, is_openimages_f16_8192: bool = False, quantize: bool = True):
    # z_mask not defined in notebook, appears to be unused
    # if constraint_regions:
    #     z = replace_grad(z, z * z_mask)

    if quantize:
        if is_openimages_f16_8192:
            z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(
                3, 1
            )
        else:
            z_q = vector_quantize(
                z.movedim(1, 3), model.quantize.embedding.weight
            ).movedim(3, 1)

    else:
        z_q = z.model

    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)
