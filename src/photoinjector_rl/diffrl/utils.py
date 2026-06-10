"""
Vendored helpers from DiffRL/utils. Bundles RunningMeanStd, CriticDataset,
AverageMeter, TimeReport, seeding(), grad_norm(), and print helpers.

The original DiffRL files (running_mean_std.py, dataset.py, average_meter.py,
time_report.py, common.py, torch_utils.py) are merged here to avoid a long
sys.path dance. No behavioural changes.
"""
# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved. Header preserved
# from the upstream files.
from __future__ import annotations

import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# print helpers (from common.py)
# ---------------------------------------------------------------------------

def print_info(*message):
    print("\033[96m", *message, "\033[0m")


# ---------------------------------------------------------------------------
# seeding (from common.py)
# ---------------------------------------------------------------------------


def seeding(seed: int = 0, torch_deterministic: bool = False) -> int:
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return seed


# ---------------------------------------------------------------------------
# grad norm (from torch_utils.py)
# ---------------------------------------------------------------------------


def grad_norm(params) -> torch.Tensor:
    total = torch.tensor(0.0)
    for p in params:
        if p.grad is not None:
            total = total + torch.sum(p.grad ** 2)
    return torch.sqrt(total)


# ---------------------------------------------------------------------------
# RunningMeanStd (from running_mean_std.py)
# ---------------------------------------------------------------------------


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (),
                 device: str = "cuda:0"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def to(self, device):
        rms = RunningMeanStd(device=device)
        rms.mean = self.mean.to(device).clone()
        rms.var = self.var.to(device).clone()
        rms.count = self.count
        return rms

    @torch.no_grad()
    def update(self, arr: torch.Tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0, unbiased=False)
        batch_count = arr.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m_2 / tot
        self.count = tot

    def normalize(self, arr: torch.Tensor, un_norm: bool = False) -> torch.Tensor:
        if not un_norm:
            return (arr - self.mean) / torch.sqrt(self.var + 1e-5)
        return arr * torch.sqrt(self.var + 1e-5) + self.mean


# ---------------------------------------------------------------------------
# CriticDataset (from dataset.py)
# ---------------------------------------------------------------------------


class CriticDataset:
    def __init__(self, batch_size, obs, target_values, shuffle=False,
                 drop_last=False):
        self.obs = obs.view(-1, obs.shape[-1])
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size
        if shuffle:
            self.shuffle()
        if drop_last:
            self.length = self.obs.shape[0] // self.batch_size
        else:
            self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1

    def shuffle(self):
        index = np.random.permutation(self.obs.shape[0])
        self.obs = self.obs[index, :]
        self.target_values = self.target_values[index]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {"obs": self.obs[start:end, :],
                "target_values": self.target_values[start:end]}


# ---------------------------------------------------------------------------
# AverageMeter (from average_meter.py)
# ---------------------------------------------------------------------------


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# TimeReport (from time_report.py)
# ---------------------------------------------------------------------------


class _Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.time_total = 0.0

    def on(self):
        assert self.start_time is None, f"Timer {self.name} already on"
        self.start_time = time.time()

    def off(self):
        assert self.start_time is not None, f"Timer {self.name} not started"
        self.time_total += time.time() - self.start_time
        self.start_time = None

    def report(self):
        print_info(f"Time report [{self.name}]: {self.time_total:.2f} s")

    def clear(self):
        self.start_time = None
        self.time_total = 0.0


class TimeReport:
    def __init__(self):
        self.timers = {}

    def add_timer(self, name):
        assert name not in self.timers
        self.timers[name] = _Timer(name)

    def start_timer(self, name):
        self.timers[name].on()

    def end_timer(self, name):
        self.timers[name].off()

    def report(self, name=None):
        if name is not None:
            self.timers[name].report()
        else:
            print_info("------------Time Report------------")
            for t in self.timers.values():
                t.report()
            print_info("-----------------------------------")

    def clear_timer(self, name=None):
        if name is not None:
            self.timers[name].clear()
        else:
            for t in self.timers.values():
                t.clear()
