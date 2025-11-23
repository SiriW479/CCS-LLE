# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unprocesses sRGB images into realistic raw data.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist


def random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
               [-0.5625, 1.6328, -0.0469],
               [-0.0703, 0.2188, 0.6406]],
              [[0.4913, -0.0541, -0.0202],
               [-0.613, 1.3513, 0.2906],
               [-0.1564, 0.2151, 0.7183]],
              [[0.838, -0.263, -0.0639],
               [-0.2887, 1.0725, 0.2496],
               [-0.0627, 0.1427, 0.5438]],
              [[0.6596, -0.2079, -0.0562],
               [-0.4782, 1.3016, 0.1933],
               [-0.097, 0.1581, 0.5181]]]
    num_ccms = len(xyz2cams)
    xyz2cams = torch.FloatTensor(xyz2cams)
    weights  = torch.FloatTensor(num_ccms, 1, 1).uniform_(1e-8, 1e8)
    weights_sum = torch.sum(weights, dim=0)
    xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                               [0.2126729, 0.7151522, 0.0721750],
                               [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    return rgb2cam


def random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    n        = tdist.Normal(loc=torch.tensor([0.8]), scale=torch.tensor([0.1])) 
    rgb_gain = 1.0 / n.sample()

    # Red and blue gains represent white balance.
    red_gain  =  torch.FloatTensor(1).uniform_(1.9, 2.4)
    blue_gain =  torch.FloatTensor(1).uniform_(1.5, 1.9)

    #   red_gain  =  torch.FloatTensor(1).uniform_(1.2, 2.4)
    #   blue_gain =  torch.FloatTensor(1).uniform_(1.2, 2.4)
    return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    image = torch.clamp(image, min=0.0, max=1.0)
    out   = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0) 
    out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out


def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    out   = torch.clamp(image, min=1e-8) ** 2.2
    out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out


def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    shape = image.size()
    image = torch.reshape(image, [-1, 3])
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    out   = torch.reshape(image, shape)
    out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    gains = torch.stack((1.0 / red_gain, torch.tensor([1.0]), 1.0 / blue_gain)) / rgb_gain
    gains = gains.squeeze()
    gains = gains[None, None, :]
    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray  = torch.mean(image, dim=-1, keepdim=True)
    inflection = 0.9
    mask  = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
    safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
    out   = image * safe_gains
    out   = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out


def mosaic(image):
    """Extracts RGGB Bayer planes from an RGB image."""
    image = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    shape = image.size()
    red   = image[0::2, 0::2, 0]
    green_red  = image[0::2, 1::2, 1]
    green_blue = image[1::2, 0::2, 1]
    blue = image[1::2, 1::2, 2]
    out  = torch.stack((red, green_red, green_blue, blue), dim=-1)
    out  = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
    out  = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out


def unprocess(image):
    """Unprocesses an image from sRGB to realistic raw data."""

    # Randomly creates image metadata.
    rgb2cam = random_ccm()
    cam2rgb = torch.inverse(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()

    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    # Approximately inverts white balance and brightening.
    #   image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    image = torch.clamp(image, min=0.0, max=1.0)
    # Applies a Bayer mosaic.
    image = mosaic(image)

    metadata = {
      'cam2rgb': cam2rgb,
      'rgb_gain': rgb_gain,
      'red_gain': red_gain,
      'blue_gain': blue_gain,
    }
    return image, metadata

def unprocess_clean(image):
    """Unprocesses an image from sRGB to realistic raw data."""
    rgb2cam = random_ccm()
    cam2rgb = torch.inverse(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()
    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    # Approximately inverts white balance and brightening.
    #   image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    image = torch.clamp(image, min=0.0, max=1.0)
    metadata = {
      'cam2rgb': cam2rgb,
      'rgb_gain': rgb_gain,
      'red_gain': red_gain,
      'blue_gain': blue_gain,
    }
    return image, metadata


def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise     = torch.FloatTensor(1).uniform_(log_min_shot_noise, log_max_shot_noise)
    #   log_shot_noise     = torch.FloatTensor([np.log(0.012)])
    shot_noise = torch.exp(log_shot_noise)
    #   print(shot_noise,shot_noise.dtype)
    #   shot_noise =torch.FloatTensor([0.012])

    line = lambda x: 2.18 * x + 1.20
    n    = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26])) 
    log_read_noise = line(log_shot_noise) + n.sample()
    read_noise     = torch.exp(log_read_noise)
    return shot_noise, read_noise

def random_noise_levels_fix(factor):
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = np.log(factor)
    log_max_shot_noise = np.log(factor)
#     log_min_shot_noise = np.log(0.0001)
#     log_max_shot_noise = np.log(0.012)
    log_shot_noise     = torch.FloatTensor(1).uniform_(log_min_shot_noise, log_max_shot_noise)
    #   log_shot_noise     = torch.FloatTensor([np.log(0.012)])
    shot_noise = torch.exp(log_shot_noise)
    #   print(shot_noise,shot_noise.dtype)
    #   shot_noise =torch.FloatTensor([0.012])

    line = lambda x: 2.18 * x + 1.20
    n    = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26])) 
    log_read_noise = line(log_shot_noise) + n.sample()
    read_noise     = torch.exp(log_read_noise)
    return shot_noise, read_noise

def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    image    = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
    variance = image * shot_noise + read_noise
#     print(torch.sqrt(variance))
    n        = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance)) 
    noise    = n.sample()
    out      = image + noise
    out      = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
    return out

def apply_gains(bayer_images, red_gains, blue_gains):
    """Applies white balance gains to a batch of Bayer images."""
    red_gains = red_gains.squeeze(1)
    blue_gains= blue_gains.squeeze(1)
    bayer_images = bayer_images.permute(0, 2, 3, 1) # Permute the image tensor to BxHxWxC format from BxCxHxW format
    green_gains  = torch.ones_like(red_gains)
    gains = torch.stack([red_gains, green_gains, green_gains, blue_gains], dim=-1)
    gains = gains[:, None, None, :]
    outs  = bayer_images * gains
    outs  = outs.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return outs


def demosaic(bayer_images):
    def SpaceToDepth_fact2(x):
        # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        bs = 2
        N, C, H, W = x.size()
        x = x.view(N, C, H // bs, bs, W // bs, bs)      # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()    # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (bs ** 2), H // bs, W // bs)  # (N, C*bs^2, H//bs, W//bs)
        return x
    def DepthToSpace_fact2(x):
        # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        bs = 2
        N, C, H, W = x.size()
        x = x.view(N, bs, bs, C // (bs ** 2), H, W)     # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()    # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (bs ** 2), H * bs, W * bs)   # (N, C//bs^2, H * bs, W * bs)
        return x

    """Bilinearly demosaics a batch of RGGB Bayer images."""
    bayer_images = bayer_images.permute(0, 2, 3, 1) # Permute the image tensor to BxHxWxC format from BxCxHxW format

    shape = bayer_images.size()
    shape = [shape[1] * 2, shape[2] * 2]

    red = bayer_images[Ellipsis, 0:1]
    upsamplebyX = nn.Upsample(size=shape, mode='bilinear', align_corners=False)
    red = upsamplebyX(red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_red = bayer_images[Ellipsis, 1:2]
    green_red = torch.flip(green_red, dims=[1]) # Flip left-right
    green_red = upsamplebyX(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    green_red = torch.flip(green_red, dims=[1]) # Flip left-right
    green_red = SpaceToDepth_fact2(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_blue = bayer_images[Ellipsis, 2:3]
    green_blue = torch.flip(green_blue, dims=[0]) # Flip up-down
    green_blue = upsamplebyX(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    green_blue = torch.flip(green_blue, dims=[0]) # Flip up-down
    green_blue = SpaceToDepth_fact2(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_at_red = (green_red[Ellipsis, 0] + green_blue[Ellipsis, 0]) / 2
    green_at_green_red = green_red[Ellipsis, 1]
    green_at_green_blue = green_blue[Ellipsis, 2]
    green_at_blue = (green_red[Ellipsis, 3] + green_blue[Ellipsis, 3]) / 2

    green_planes = [
      green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = DepthToSpace_fact2(torch.stack(green_planes, dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    blue = bayer_images[Ellipsis, 3:4]
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])
    blue = upsamplebyX(blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])

    rgb_images = torch.cat([red, green, blue], dim=-1)
    rgb_images = rgb_images.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return rgb_images


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(0, 2, 3, 1) # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms   = ccms[:, None, None, :, :]
    outs   = torch.sum(images * ccms, dim=-1)
    outs   = outs.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    # Clamps to prevent numerical instability of gradients near zero.
    images = images.permute(0, 2, 3, 1) # Permute the image tensor to BxHxWxC format from BxCxHxW format
    outs   = torch.clamp(images, min=1e-8) ** (1.0 / gamma)
    outs   = outs.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return outs

def smoothstep(img):
    smooth=3*img**2-2*img**3
    return smooth.clamp(min=0.0,max=1.0)


def process(bayer_images, red_gains, blue_gains, cam2rgbs):
    """Processes a batch of Bayer RGGB images into sRGB images."""
    # White balance.
    #   bayer_images = apply_gains(bayer_images, red_gains, blue_gains)
    # Demosaic.
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = demosaic(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images)
    images = smoothstep(images)
    return images

def process_clean(bayer_images, red_gains, blue_gains, cam2rgbs):
    """Processes a batch of Bayer RGGB images into sRGB images."""
    # White balance.
    #   bayer_images = apply_gains(bayer_images, red_gains, blue_gains)
    # Demosaic.
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = bayer_images
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images)
    images = smoothstep(images)
    return images


class addISPNoise(object):
    def __init__(self,factor):
        assert isinstance(factor,float)
        self.factor = factor


    def __call__(self, sample ):
        image_in_img, metadata=unprocess(sample)
        shot_noise, read_noise = random_noise_levels()
        noisy_img= add_noise(image_in_img, self.factor*shot_noise, read_noise)
        rgb_image_in  = process(noisy_img.unsqueeze(0), metadata['red_gain'].unsqueeze(1), metadata['blue_gain'].unsqueeze(1), metadata['cam2rgb'].unsqueeze(1)).squeeze(1)
        out = rgb_image_in.clamp(min = 0,max =1)
        return out

class addISPNoise_clean(object):
    def __init__(self,factor):
        assert isinstance(factor,float)
        self.factor = factor


    def __call__(self, sample ):
        image_in_img, metadata=unprocess_clean(sample)
        shot_noise, read_noise = random_noise_levels_fix(self.factor)
        noisy_img= add_noise(image_in_img, shot_noise, read_noise)
        rgb_image_in  = process_clean(noisy_img.unsqueeze(0), metadata['red_gain'].unsqueeze(1), metadata['blue_gain'].unsqueeze(1), metadata['cam2rgb'].unsqueeze(1)).squeeze(1)
        out = rgb_image_in.clamp(min = 0,max =1)
        return out    
    