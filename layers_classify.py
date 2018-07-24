import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from .utils import check_or_create_dir
from helpers.layers import View, add_normalization
from helpers.distributions import nll_activation as nll_activation_fn
from helpers.distributions import nll as nll_fn



def build_dense_classifier(input_shape, output_size, latent_size=512, activation_fn=nn.ELU, normalization_str="none"):
    input_flat = int(np.prod(input_shape))
    output_flat = int(np.prod(output_size))
    return nn.Sequential(
        View([-1, input_flat]),
        nn.Linear(input_flat, latent_size),
        add_normalization(normalization_str, 1, latent_size, num_groups=32),
        activation_fn(),
        nn.Linear(latent_size, latent_size),
        add_normalization(normalization_str, 1, latent_size, num_groups=32),
        activation_fn(),
        nn.Linear(latent_size, output_flat),
        View([-1] + number_labels)
    )