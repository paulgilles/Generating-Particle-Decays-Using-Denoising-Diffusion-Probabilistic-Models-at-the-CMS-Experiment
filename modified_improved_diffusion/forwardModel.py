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

def reshape_on_gpu(tensor, shape):
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
        out_channels,
        emb_layer,
        emb_nodes,
        structure,
        dropout=0,
        norm_layer=None,
        dropout_layer=None,
        num_classes=None,
        use_checkpoint=False,
        use_scale_shift_norm=False
    ):
        super().__init__()

        self.out_channels=out_channels
        self.dropout=dropout
        self.num_classes=num_classes
        self.use_checkpoint=use_checkpoint
        self.use_scale_shift_norm=use_scale_shift_norm
        self.emb_layer=emb_layer
        self.structure=structure
        self.emb_nodes=emb_nodes
        self.norm_layer=norm_layer
        self.dropout=dropout
        self.dropout_layer=dropout_layer

        self.ReLU =  th.nn.ReLU()
        self.softmax = th.nn.Softmax(dim=1)
        self.tanh = th.nn.Tanh()

        if self.emb_nodes % 4 != 0:
            raise ValueError("‘emb_nodes‘ needs to be ∈ {i * 4 | i ∈ ℕ}")

        self.time_embed = nn.Sequential(
            linear(int(self.emb_nodes/4), self.emb_nodes),
            SiLU(),
            linear(self.emb_nodes, self.emb_nodes),
        )

    #@note Hier fehlt ein Klassenembedding, siehe UNet

        self.network= self.creating_network(self.out_channels, self.structure,
                                            self.emb_layer, self.norm_layer,
                                            self.dropout, self.dropout_layer)


    def forward(self, x, timesteps, y=None):

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        emb = self.time_embed(timestep_embedding(timesteps, int(self.emb_nodes/4)))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        microbatches = np.shape(x)[0]
        
        h = x #@note Hier muss man noch die Kompabilität mit anderen types einbauen
        h = reshape_on_gpu(h, shape=(microbatches,8))
        h = self.network(h, emb)
        h = reshape_on_gpu(h, shape=(microbatches, self.out_channels, 4))
        h = h.type(x.dtype)
        return h
    

    def creating_network(self, out_channels, structure, emb_layer, 
                         norm_layer=None, dropout=0, dropout_layer=None):
        network = []
        for index, nodes in enumerate(structure):
            if index == 0:
                network += [nn.Linear(8, nodes)]
            else: 
                prev_nodes = structure[index-1]
                network += [nn.Linear(prev_nodes, nodes)]
            if index in emb_layer:
                network += [EmbeddingBlock(self.emb_nodes, nodes)]
            if norm_layer is not None:
                if index in norm_layer:
                    network += [nn.GroupNorm(int(nodes/2), nodes)]
            if dropout_layer is not None:
                if index in dropout_layer:
                    network += [nn.Dropout(p=dropout)]
            network += [self.ReLU]
        prev_nodes = structure[-1]
        network += [nn.Linear(prev_nodes, out_channels*4)]
        return TimestepEmbedSequential(*network)



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """   


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class EmbeddingBlock(TimestepBlock):
    def __init__(self, emb_nodes, timestepBlock_output):
        super().__init__()

        self.emb_layers = nn.Sequential(
            linear(emb_nodes, timestepBlock_output),
            SiLU(),
            linear(timestepBlock_output, timestepBlock_output)
        )

    def forward(self, x, emb):
        emb_out = self.emb_layers(emb).type(x.dtype)
        x = x + emb_out
        return x
    
