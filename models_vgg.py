from typing import Callable
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms


# Core dependencies
import jax
import jax.numpy as jnp
import optax

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf
from omegacli import OmegaConf
# stune
import stune
import json
import copy

VGG_types = {
    "CNN": [64, "M", 128, "M", 128, 64, "M"],
    "EP": [128, "M", 256, "M", 512, "M", 512, "M"],
    "VGG9": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"],
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],

}

class VGG5(pxc.EnergyModule):
    def __init__(
        self,
        nm_classes: int,
        input_size: int,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool,
    ) -> None:
        super().__init__()

        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)

        self.feature_layers = [
            (
                pxnn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(128, 256, kernel_size=(3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            )
        ]
        self.classifier_layers = [
            (
                pxnn.Linear(512 * (input_size//16) * (input_size//16), self.nm_classes.get()),
            ),
        ]

        self.vodes = [
            pxc.Vode() for _ in range(len(self.feature_layers))] + [
            pxc.Vode(pxc.se_energy if se_flag else pxc.ce_energy)]
        self.vodes[-1].h.frozen = True
        for i in range(len(self.vodes)):
            self.vodes[i].h0.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array, out_key="h", beta: float = 1.0, inference: bool = True, update_state: bool = True):
        for block, node in zip(self.feature_layers, self.vodes[:len(self.feature_layers)]):
            for layer in block:
                x = layer(x)
            x = node(x, output=out_key)

        x = x.flatten()
        for block, node in zip(self.classifier_layers, self.vodes[len(self.feature_layers):]):
            for layer in block:
                x = layer(x)
            x = node(x, output=out_key)
        
        if y is not None:
            self.vodes[-1].set("h", self.vodes[-1].get("u") - beta * (self.vodes[-1].get("u") - y))
           
        return self.vodes[-1].get("u")
    
class VGG7(pxc.EnergyModule):
    def __init__(
        self,
        nm_classes: int,
        input_size: int,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool,
    ) -> None:
        super().__init__()

        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)

        self.feature_layers = [
            (
                pxnn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                self.act_fn
            ),
            (
                pxnn.Conv2d(128, 256, kernel_size=(3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(256, 256, kernel_size=(3, 3), padding=(0, 0)),
                self.act_fn
            ),
            (
                pxnn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1)),
                self.act_fn,
                pxnn.MaxPool2d(kernel_size=2, stride=2)
            ),
            (
                pxnn.Conv2d(512, 512, kernel_size=(3, 3), padding=(0, 0)),
                self.act_fn
            )
        ]
        self.classifier_layers = [
            (
                pxnn.Linear(512 * ((input_size//4 - 2)//2 - 2) * ((input_size//4 - 2)//2 - 2), self.nm_classes.get()),
            ),
        ]

        self.vodes = [
            pxc.Vode() for _ in range(len(self.feature_layers))] + [
            pxc.Vode(pxc.se_energy if se_flag else pxc.ce_energy)]
        self.vodes[-1].h.frozen = True
        for i in range(len(self.vodes)):
            self.vodes[i].h0.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array, out_key="h", beta: float = 1.0, inference: bool = True, update_state: bool = True):
        for block, node in zip(self.feature_layers, self.vodes[:len(self.feature_layers)]):
            for layer in block:
                x = layer(x)
            x = node(x, output=out_key)

        x = x.flatten()
        for block, node in zip(self.classifier_layers, self.vodes[len(self.feature_layers):]):
            for layer in block:
                x = layer(x)
            x = node(x, output=out_key)

        if y is not None:
            self.vodes[-1].set("h", self.vodes[-1].get("u") - beta * (self.vodes[-1].get("u") - y))
           
        return self.vodes[-1].get("u")

class VGGNet_Skip(pxc.EnergyModule):
    def __init__(
        self,
        nm_classes: int,
        in_height: int,
        in_width: int,
        in_channels: int,
        model_type: str,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool
    ) -> None:
        super().__init__()
        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)
        self.se_flag = se_flag
        
        self.feature_layers, vodes_feature, self.long_skip_2 = self.init_convs(VGG_types[model_type], in_channels, in_height, in_width)
        self.classifier_layers, vodes_classifer = self.init_fcs(VGG_types[model_type], in_height, in_width, 4096, self.nm_classes.get())
        self.vodes = vodes_feature + vodes_classifer
        self.vodes[-1].h.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array = None, beta: float = 1.0, out_key='h'):
        for ind, (block, node) in enumerate(zip(self.feature_layers, self.vodes[: len(self.feature_layers)])):
            for layer in block:
                x = layer(x)
            x = node(x, output=out_key)
            # if ind == 2:
            #     long_skip_value_1 = self.long_skip_1(x.flatten())
            if ind == 7:
                long_skip_value_2 = self.long_skip_2(x.flatten())
            # if ind == 11:
            #     long_skip_value_3 = self.long_skip_3(x.flatten())
        x = x.flatten()
        for ind, (block, node) in enumerate(zip(self.classifier_layers, self.vodes[len(self.feature_layers) :])):
            for layer in block:
                x = layer(x)
            if ind == len(self.classifier_layers) - 1:
                x = x + long_skip_value_2 # + long_skip_value_3
            x = node(x)
        if y is not None:
            self.vodes[-1].set("h", self.vodes[-1].get("u") - beta * (self.vodes[-1].get("u") - y))
        return self.vodes[-1].get("u")
    
    def init_convs(self, architecture, in_channels, in_height, in_width):
        layers = []
        vodes = []
        for i in range(len(architecture) - 1):
            x = architecture[i]
            next_x = architecture[i + 1]
            if type(x) == int:
                out_channel = x
                if type(next_x) == int:
                    layers.append(
                        (
                            pxnn.Conv2d(in_channels, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            self.act_fn,
                        )
                    )
                    vodes.append(
                        pxc.Vode()
                    )
                elif next_x == "M":
                    layers.append(
                        (
                            pxnn.Conv2d(in_channels, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            self.act_fn,
                            pxnn.MaxPool2d(kernel_size=2, stride=2)
                        )
                    )
                    in_height = in_height // 2
                    in_width = in_width // 2
                    vodes.append(
                        pxc.Vode()
                    )
                else:
                    raise ValueError(
                        f"some errors in architecture file"
                    )
                in_channels = x
                # if i == 4:
                #    long_skip_1 = pxnn.Linear(128 * 16 * 16, self.nm_classes.get())
                if i == 9:
                    long_skip_2 = pxnn.Linear(256 * 4 * 4, self.nm_classes.get())
                # if i == 14:
                #     long_skip_3 = pxnn.Linear(512 * 2 * 2, self.nm_classes.get())
            else:
                pass
        return layers, vodes, long_skip_2
        
    def init_fcs(self, architecture, in_height, in_width, num_hidden, nm_classes):
        # print(in_height, in_width)
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (in_height % factor) + (in_width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )
        out_height = in_height // factor
        out_width = in_width // factor
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )
        layers = [
            (
                pxnn.Linear(last_out_channels * out_height * out_width, num_hidden),
                self.act_fn,
                # Mark, do we need a DropOut?
            ),
            (
                pxnn.Linear(num_hidden, num_hidden),
                self.act_fn,
            ),
            (
                pxnn.Linear(num_hidden, nm_classes),
                # self.act_softmax
            ),
        ]
        vodes = [
            pxc.Vode(),
            pxc.Vode(),
            pxc.Vode(pxc.se_energy) if self.se_flag else pxc.Vode(pxc.ce_energy)
        ]
        return layers, vodes
    
class VGGNet(pxc.EnergyModule):
    def __init__(
        self,
        nm_classes: int,
        in_height: int,
        in_width: int,
        in_channels: int,
        model_type: str,
        act_fn: Callable[[jax.Array], jax.Array],
        se_flag: bool
    ) -> None:
        super().__init__()
        self.nm_classes = px.static(nm_classes)
        self.act_fn = px.static(act_fn)
        self.se_flag = se_flag
        
        self.feature_layers, vodes_feature = self.init_convs(VGG_types[model_type], in_channels, in_height, in_width)
        self.classifier_layers, vodes_classifer = self.init_fcs(VGG_types[model_type], in_height, in_width, 4096, self.nm_classes.get())
        self.vodes = vodes_feature + vodes_classifer
        self.vodes[-1].h.frozen = True
        for i in range(len(self.vodes)):
            self.vodes[i].h0.frozen = True

    def __call__(self, x: jax.Array, y: jax.Array = None, out_key='h', beta: float = 1.0, inference: bool = True, update_state: bool = True):
        for _, (block, node) in enumerate(zip(self.feature_layers, self.vodes[: len(self.feature_layers)])):
            for layer in block:
                x = layer(x)
                # print(x.shape)
            x = node(x, output=out_key)
        x = x.flatten()
        for _, (block, node) in enumerate(zip(self.classifier_layers, self.vodes[len(self.feature_layers) :])):
            for layer in block:
                x = layer(x)
                # print(x.shape)
            x = node(x, output=out_key)
        if y is not None:
            self.vodes[-1].set("h", self.vodes[-1].get("u") - beta * (self.vodes[-1].get("u") - y))
        return self.vodes[-1].get("u")
    
    def init_convs(self, architecture, in_channels, in_height, in_width):
        layers = []
        vodes = []
        for i in range(len(architecture) - 1):
            x = architecture[i]
            next_x = architecture[i + 1]
            if type(x) == int:
                out_channel = x
                if type(next_x) == int:
                    layers.append(
                        (
                            pxnn.Conv2d(in_channels, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            self.act_fn,
                        )
                    )
                    vodes.append(
                        pxc.Vode()
                    )
                elif next_x == "M":
                    layers.append(
                        (
                            pxnn.Conv2d(in_channels, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            self.act_fn,
                            pxnn.MaxPool2d(kernel_size=2, stride=2)
                        )
                    )
                    in_height = in_height // 2
                    in_width = in_width // 2
                    vodes.append(
                        pxc.Vode()
                    )
                else:
                    raise ValueError(
                        f"some errors in architecture file"
                    )
                in_channels = x
            else:
                pass
        return layers, vodes
        
    def init_fcs(self, architecture, in_height, in_width, num_hidden, nm_classes):
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (in_height % factor) + (in_width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )
        out_height = in_height // factor
        out_width = in_width // factor
        # print(out_height, out_width)
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )
        layers = [
            (
                pxnn.Linear(last_out_channels * out_height * out_width, num_hidden),
                self.act_fn,
            ),
            (
                pxnn.Linear(num_hidden, num_hidden),
                self.act_fn,
            ),
            (
                pxnn.Linear(num_hidden, nm_classes),
            ),
        ]
        vodes = [
            pxc.Vode(),
            pxc.Vode(),
            pxc.Vode(pxc.se_energy) if self.se_flag else pxc.Vode(pxc.ce_energy)
        ]
        return layers, vodes
    

def get_model(
        model_name:str,
        nm_classes: int,
        act_fn: Callable[[jax.Array], jax.Array],
        input_size: int,
        se_flag: bool,
        is_skip: bool = False
):
    if model_name == "VGG5":
        return VGG5(nm_classes, input_size, act_fn, se_flag)
    elif model_name == "VGG7":
        return VGG7(nm_classes, input_size, act_fn, se_flag)
    else:
        return VGGNet(nm_classes, input_size, input_size, 3, model_name, act_fn, se_flag)