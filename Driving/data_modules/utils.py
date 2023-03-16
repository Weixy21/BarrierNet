import numpy as np
import torch
import torchvision.transforms.functional as TVF


def transform_rgb(img, train, use_color_jitter=True, use_fixed_standardize=False,
                  gamma_range=[0.5, 1.5], brightness_range=[0.5, 1.5], contrast_range=[0.3, 1.7],
                  saturation_range=[0.5, 1.5], gamma=None, brightness=None, contrast=None,
                  saturation=None):
    if not isinstance(img, torch.Tensor):
        img = TVF.to_tensor(img)
    if train and use_color_jitter:  # perform color jitter
        if gamma is None:
            gamma = np.random.uniform(*gamma_range)
        if brightness is None:
            brightness = np.random.uniform(*brightness_range)
        if contrast is None:
            contrast = np.random.uniform(*contrast_range)
        if saturation is None:
            saturation = np.random.uniform(*saturation_range)
        img = TVF.adjust_gamma(img, gamma)
        img = TVF.adjust_brightness(img, brightness)
        img = TVF.adjust_contrast(img, contrast)
        img = TVF.adjust_saturation(img, saturation)
    if use_fixed_standardize:
        img = fixed_standardize(img)
    else:
        img = per_image_standardize(img)
    return img, [gamma, brightness, contrast, saturation]


def per_image_standardize(x):
    # follow https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
    #        https://discuss.pytorch.org/t/per-image-normalization/22141/4
    mean = torch.mean(x, dim=(-1, -2))[:, None, None]
    stddev = torch.std(x, dim=(-1, -2))[:, None, None]
    num_pixels = torch.tensor(torch.numel(x), dtype=torch.float32)
    min_stddev = torch.rsqrt(num_pixels)
    adjusted_stddev = torch.max(stddev, min_stddev)
    return (x - mean) / adjusted_stddev


def fixed_standardize(x):
    mean = torch.tensor([0.3409, 0.3083, 0.2384]).to(x)[:, None, None]
    std = torch.tensor([0.2436, 0.1835, 0.1901]).to(x)[:, None, None]
    return (x - mean) / std
