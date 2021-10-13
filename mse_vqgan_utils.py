import torch
from torch import nn
import torch.nn.functional as F
import sys

from vqgan_utils import resample, clamp_with_grad, vector_quantize

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

        # min_size_width = min(sideX, sideY)

        for _ in range(self.cutn):

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
