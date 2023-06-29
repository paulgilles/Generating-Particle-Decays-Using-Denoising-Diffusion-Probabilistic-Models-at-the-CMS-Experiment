from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modified_improved_diffusion.debugging import debug_mode, debug_print
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
import modified_improved_diffusion.dist_util as dist_util

def reshape_from_gpu(tensor, shape):
    """
    It pushes a tensor to the CPU, reshapes it and then pushes is back 
    """
    tensor = tensor.cpu()
    tensor = tensor.reshape(shape)
    tensor = tensor.to(dist_util.dev())
    return tensor


class forwardNet(nn.Module):
    def __init__(
        self,
        model_channels,
        out_channels,
        dropout=0,
        num_classes=None,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        emb_layer=-1
    ):
        super().__init__()

        self.model_channels=model_channels
        self.out_channels=out_channels
        self.dropout=dropout
        self.num_classes=num_classes
        self.use_checkpoint=use_checkpoint
        self.use_scale_shift_norm=use_scale_shift_norm
        self.emb_layer=emb_layer

        self.hidden1 = nn.Linear(8, 6)
        self.hidden2 = nn.Linear(6, 6)
        self.hidden3 = nn.Linear(6, 8)
        self.output = nn.Linear(8, out_channels * 4)
        self.activation = th.nn.ReLU()
        self.softmax = th.nn.Softmax(dim=1)

        self.network = [
            self.hidden1,
            self.hidden2,
            self.hidden3,
            self.output
        ]

        time_embed_dim = 3 * model_channels
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )


    def forward(self, x, timesteps, y=None):

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        #h = x.type(self.inner_dtype)   
        h = x
        h = reshape_from_gpu(h, shape=(16,8))
        if self.emb_layer != "all":
            if self.emb_layer >= len(self.network)-3:
                raise ValueError((f"Das Netzwerk besteht nur aus {len(self.network)-1}"
                                    "hidden Layers. Embedding Layer kann nur in"
                                f"[0, {len(self.network)-3}] liegen. Nutze -1"
                                    "für Embedding in jeder Schicht."))
        for index, layer in enumerate(self.network):
            h = layer(h)
            if layer != self.output:
                h = self.activation(h)
                if layer != self.hidden3:
                    if self.emb_layer == -1:
                        h = h + emb
                    else:
                        if index == self.emb_layer:
                            h = h + emb
            else: 
                h = self.softmax(h)
        h = reshape_from_gpu(h, shape=(16, self.out_channels, 4))
        h = h.type(x.dtype)
        return h



