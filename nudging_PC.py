from typing import Callable
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

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
import gc

# stune
import stune
import json

import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models_vgg import get_model
from datasets import get_loader

def get_datasetinfo(dataset, model_name=None):
    if dataset == "MNIST":
        return 10, 28
    elif dataset == "FashionMNIST":
        return 10, 28
    elif dataset == "CIFAR10":
        return 10, 32
    elif dataset == "CIFAR100":
        return 100, 32
    elif dataset == "TinyImageNet":
        if model_name in ['VGG5', 'VGG7']:
            return 200, 56
        else:
            return 200, 64
    else:
        raise ValueError("Invalid dataset name")

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def generate_sequence(x, k=1.0):
    """
    生成一个长度为 x 的数列，数列和为 1，且曲线一开始变化快，后面变平缓。
    
    参数：
        x: 数列长度
        k: 衰减速度参数，越大曲线变化越快（默认值为 1.0）
    
    返回：
        一个长度为 x 的数列，和为 1
    """
    # 生成指数衰减数列
    indices = np.arange(x)  # 生成 0 到 x-1 的索引
    raw_sequence = np.exp(-k * indices)  # 应用指数衰减
    # 归一化处理，使数列和为 1
    normalized_sequence = raw_sequence / np.sum(raw_sequence)
    return normalized_sequence

def get_sequences(N, T, k):
    """
    生成 N 个长度为 T 的数列。
    
    参数：
        N: 要生成的数列数量
        T: 每个数列的长度
        k: 衰减速度参数
    
    返回：
        一个 N*T 的矩阵，包含 N 个数列
    """
    sequences = np.zeros((N, T))
    for i in range(N):
        sequences[N-i-1, i:T] = generate_sequence(T-i, k)
    return sequences.T.tolist()

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0), out_axes=0, axis_name="batch")
def forward(x, y, *, model, beta=1.0):
    return model(x, y, beta=beta)

import functools

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energy(x, *, model, weights):
    y_ = model(x, None)
    energies = [m.energy() for m in model.submodules(cls=pxc.EnergyModule)]
    weighted_energy = [w * e for w, e in zip(weights, energies)]
    total_energy = functools.reduce(lambda x, y: x + y, weighted_energy)
    return jax.lax.psum(total_energy, "batch"), y_

@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0,), out_axes=(None, 0), axis_name="batch")
def energyW(x, *, model ):
    y_ = model(x, None, out_key="h0")
    return jax.lax.psum(model.energy(), "batch"), y_


@pxf.jit(static_argnums=0, donate_argnames=("model", "optim"))
def train_on_batch(T: int, x: jax.Array, y: jax.Array, *, model , optim_w: pxu.Optim, optim_h: pxu.Optim, weight_list, beta: float = 1.0):
    model.train()

    # nudging with beta

    # Init step
    with pxu.step(model, (pxc.STATUS.INIT, None), clear_params=pxc.VodeParam.Cache):
        logits = forward(x, y, model=model, beta=beta)

    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    for t in range(T):
        def energyN(x, model, ind=t):
            return energy(x, model=model, weights=weight_list[ind])
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            mask = pxu.M_hasnot(pxc.VodeParam, frozen=True).to([False, True])(model, is_pytree=True)
            _, g = pxf.value_and_grad({"model": mask}, has_aux=True)(
                energyN
            )(x, model=model)
        optim_h.step(model, g["model"])
           
    optim_h.clear()
    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        (e, _), g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energyW)(
            x, model=model
        )
    u = optim_w.step(model, g["model"], scale_by=1.0 / x.shape[0], apply_updates=False)
    u = jax.tree_util.tree_map(lambda x: x/beta, u)
    optim_w.apply_updates(model, u)

    return logits, e / x.shape[0]
    
    
    
@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache | pxc.VodeParam):
        outputs = forward(x, None, model=model)
    top1_pred = outputs.argmax(axis=-1)
    top5_indices = jax.lax.top_k(outputs, k=5)[1]

    top1_acc = (top1_pred == y).mean()
    top5_acc = jnp.any(top5_indices == y[:, None], axis=-1).mean()

    return top1_acc, top5_acc, top1_pred


def train(dl, T, *, model , optim_w: pxu.Optim, optim_h: pxu.Optim, weight_list, beta: float = 1.0):
    for i, (x, y) in enumerate(dl):
        train_on_batch(
            T, x, jax.nn.one_hot(y, model.nm_classes.get()), model=model, optim_w=optim_w, optim_h=optim_h, weight_list=weight_list, beta=beta
        )


def eval(dl, *, model):
    acc = []
    acc5 = []
    ys_ = []

    for x, y in dl:
        a, a5, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        acc5.append(a5)
        ys_.append(y_)

    return np.mean(acc), np.mean(acc5), np.concatenate(ys_)


def main(run_info: stune.RunInfo, save_model: bool):

    if save_model:
        save_name = os.path.join('~/model_weights_h0', os.path.split(run_info["study"])[1])

   
    dataset_name = run_info["hp/dataset"]

    batch_size = run_info["hp/batch_size"]
    nm_epochs = run_info["hp/epochs"]

    nm_classes, input_size = get_datasetinfo(dataset_name, run_info["hp/model"])

    model = get_model(
        model_name=run_info["hp/model"], 
        nm_classes=nm_classes, 
        act_fn=getattr(jax.nn, run_info["hp/act_fn"]),
        input_size=input_size,
        se_flag=run_info["hp/se_flag"])
    
    weight_list = get_sequences(len(model.vodes), run_info["hp/T"], run_info["hp/k"])

    train_dataloader, test_dataloader = get_loader(dataset_name, batch_size, input_size)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=run_info["hp/optim/w/lr"],
        peak_value=1.1 * run_info["hp/optim/w/lr"],
        warmup_steps=0.1 * len(train_dataloader) * nm_epochs,
        decay_steps=len(train_dataloader)*nm_epochs,
        end_value=0.1 * run_info["hp/optim/w/lr"],
        exponent=1.0)

    optim_h = pxu.Optim(
            optax.sgd(run_info["hp/optim/x/lr"], momentum=run_info["hp/optim/x/momentum"]),
    )
    optim_w = pxu.Optim(optax.adamw(schedule, weight_decay=run_info["hp/optim/w/wd"]), pxu.M(pxnn.LayerParam)(model))
    
    best_accuracy = 0
    best_accuracy5 = 0
    accuracies = []
    accuracies5 = []
    # if run_info["hp/beta_factor"] == 0:
    if os.path.exists('../' + run_info['study'][10:-4] + 'beta.json'):
        with open('../' + run_info['study'][10:-4] + 'beta.json', 'r') as file:
            betalist = json.load(file)
            print(betalist)
    else:
        betalist = []
    below_times = 0
    best_times = 0
    for e in range(50):
        # if run_info["hp/beta_factor"] == 0:
        try:
            beta = betalist[e]
        except:
            if run_info["hp/beta_factor"] == 0:
                beta = random.choice([-1, 1]) * run_info["hp/beta"]
            else:
                beta = run_info["hp/beta_factor"] * run_info["hp/beta"]

        train(train_dataloader, T=run_info["hp/T"], model=model, optim_w=optim_w, optim_h=optim_h, beta=beta, weight_list=weight_list)
        a, a5, y = eval(test_dataloader, model=model)
        if e > 15 and float(a) < 0.1:
            below_times += 1
        else:
            below_times = 0
        print(a,a5)
        
        if a > best_accuracy:
            best_times = 0
            best_accuracy = a
            if save_model:
                pxu.save_params(model, save_name)
                print("saved")
        else:
            best_times += 1
        if a5 > best_accuracy5:
            best_accuracy5 = a5
        accuracies.append(float(a))
        accuracies5.append(float(a5))
        if below_times >= 5 or best_times >= 10:
            break

    del train_dataloader
    del test_dataloader
    gc.collect()

    return float(best_accuracy), float(best_accuracy5), accuracies, accuracies5


if __name__ == "__main__":
    import os
    import sys
    import seed5 as seed
    run_info = stune.RunInfo(
        OmegaConf.load(sys.argv[1])
    )
    seed.run(main)(run_info)
