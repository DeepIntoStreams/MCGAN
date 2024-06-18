# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F

'''
Compatible with MC algorithm
x_mc is Tensor of shape (MC*B,C,H,W)
'''
def DiffAugment(x, policy='', channels_first=True,x_mc=None,mc_size=10,mc=False):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        if mc:
        #reshape x_mc to (MC,B,C,H,W)
            x_mc=x_mc.reshape(mc_size,x.shape[0],x.shape[1],x.shape[2],x.shape[3])
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                if mc:
                  x,x_mc = f(x,x_mc=x_mc,mc=mc)
                else:
                  x= f(x,x_mc=x_mc,mc=mc)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    if mc:
      #reshape x_mc back into (MC*B,C,H,W)
      x_mc=x_mc.contiguous()
      x_mc=x_mc.reshape(mc_size*x.shape[0],x.shape[1],x.shape[2],x.shape[3])
      return x,x_mc
    else:
      return x

def rand_brightness(x,x_mc=None,mc=False):
    factor=(torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    x = x + factor
    if mc:
      x_mc=x_mc+factor[None,:]
      return x,x_mc
    else:
      return x


def rand_saturation(x,x_mc=None,mc=False):
    x_mean = x.mean(dim=1, keepdim=True)# (B,1,H,W)
    factor=torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2#shape (B,1,1,1)
    x = (x - x_mean) * factor + x_mean
    if mc:
      x_mc = (x_mc- x_mean[None,:]) * factor[None,:] + x_mean[None,:]
      return x,x_mc
    else:
      return x


def rand_contrast(x,x_mc=None,mc=False):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True) # (B,1,1,1)
    factor=torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5 #shape (B,1,1,1)
    x = (x - x_mean) * factor  + x_mean
    if mc:
      x_mc = (x_mc- x_mean[None,:]) * factor[None,:] + x_mean[None,:]
      return x,x_mc
    else:
      return x


def rand_translation(x,x_mc=None,mc=False,ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.25,x_mc=None,mc=False):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}