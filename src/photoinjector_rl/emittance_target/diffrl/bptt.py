"""
Full-episode BPTT (no critic), vendored from NVlabs/DiffRL/algorithms/bptt.py.

Differences from upstream:
  1. Removed dflex / envs imports; env_fn is supplied by the caller.
  2. Removed the `optim.gd` GD optimizer support — only Adam is used.
  3. Tensorboard via `torch.utils.tensorboard.SummaryWriter`.
  4. step_metrics_hook callback for external CSV pumping.

Algorithm code (full-trajectory backprop, no critic, gamma-discounted return)
is unchanged.
"""
# Copyright (c) 2022 NVIDIA CORPORATION. Header preserved from upstream.
from __future__ import annotations

import copy
import os
import time
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from . import models as _models
from .utils import (
    AverageMeter,
    RunningMeanStd,
    TimeReport,
    grad_norm,
    print_info,
    seeding,
)


class BPTT:
    def __init__(self, cfg: dict, env_fn: Callable):
        seeding(cfg["params"]["general"]["seed"])
        diff_env_cfg = cfg["params"]["diff_env"]
        self.env = env_fn(
            num_envs=cfg["params"]["config"]["num_actors"],
            device=cfg["params"]["general"]["device"],
            render=cfg["params"]["general"]["render"],
            seed=cfg["params"]["general"]["seed"],
            episode_length=diff_env_cfg.get("episode_length", 250),
            stochastic_init=diff_env_cfg.get("stochastic_env", False),
            MM_caching_frequency=diff_env_cfg.get("MM_caching_frequency", 1),
            no_grad=False,
        )
        print("num_envs =", self.env.num_envs)
        print("num_actions =", self.env.num_actions)
        print("num_obs =", self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length
        self.device = cfg["params"]["general"]["device"]
        self.gamma = cfg["params"]["config"].get("gamma", 0.99)
        self.steps_num = cfg["params"]["config"]["steps_num"]
        self.max_epochs = cfg["params"]["config"]["max_epochs"]
        self.actor_lr = float(cfg["params"]["config"]["actor_learning_rate"])
        self.lr_schedule = cfg["params"]["config"].get("lr_schedule", "linear")

        self.obs_rms: Optional[RunningMeanStd] = None
        if cfg["params"]["config"].get("obs_rms", False):
            self.obs_rms = RunningMeanStd(shape=(self.num_obs,),
                                          device=self.device)
        self.rew_scale = cfg["params"]["config"].get("rew_scale", 1.0)
        self.name = cfg["params"]["config"].get("name", "Photoinjector")
        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]
        self.grad_norm_max = cfg["params"]["config"]["grad_norm"]
        self.step_metrics_hook: Optional[Callable[[int, float, float], None]] = None

        if cfg["params"]["general"]["train"]:
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok=True)
            save_cfg = copy.deepcopy(cfg)
            if "general" in save_cfg["params"]:
                deleted = [k for k in save_cfg["params"]["general"]
                           if k in save_cfg["params"]["config"]]
                for k in deleted:
                    del save_cfg["params"]["general"][k]
            with open(os.path.join(self.log_dir, "cfg.yaml"), "w") as f:
                yaml.dump(save_cfg, f)
            self.writer = SummaryWriter(os.path.join(self.log_dir, "log"))
            self.save_interval = cfg["params"]["config"].get("save_interval", 500)
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not (
                cfg["params"]["config"]["player"].get("determenistic", False)
                or cfg["params"]["config"]["player"].get("deterministic", False))
            self.steps_num = self.env.episode_length

        self.actor_name = cfg["params"]["network"].get("actor", "ActorStochasticMLP")
        actor_fn = getattr(_models, self.actor_name)
        self.actor = actor_fn(self.num_obs, self.num_actions,
                              cfg["params"]["network"], device=self.device)

        if cfg["params"]["general"]["train"]:
            self.save("init_policy")

        betas = cfg["params"]["config"].get("betas", [0.7, 0.95])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                betas=betas, lr=self.actor_lr)

        self.iter_count = 0
        self.step_count = 0

        self.episode_length_his: list[int] = []
        self.episode_loss_his: list[float] = []
        self.episode_discounted_loss_his: list[float] = []
        self.episode_loss = torch.zeros(self.num_envs, dtype=torch.float32,
                                        device=self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs,
                                                   dtype=torch.float32,
                                                   device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32,
                                        device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long,
                                        device=self.device)
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf

        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        self.time_report = TimeReport()
        self.grad_norm_before_clip = torch.tensor(0.0)
        self.grad_norm_after_clip = torch.tensor(0.0)

    def compute_actor_loss(self, deterministic: bool = False) -> torch.Tensor:
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs),
                              dtype=torch.float32, device=self.device)
        gamma = torch.ones(self.num_envs, dtype=torch.float32,
                           device=self.device)
        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)

        obs = self.env.initialize_trajectory()
        if self.obs_rms is not None:
            with torch.no_grad():
                self.obs_rms.update(obs)
            obs = obs_rms.normalize(obs)

        for i in range(self.steps_num):
            actions = self.actor(obs, deterministic=deterministic)
            obs, rew, done, extra_info = self.env.step(torch.tanh(actions))
            with torch.no_grad():
                raw_rew = rew.clone()
            rew = rew * self.rew_scale
            if self.obs_rms is not None:
                with torch.no_grad():
                    self.obs_rms.update(obs)
                obs = obs_rms.normalize(obs)

            self.episode_length += 1
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                actor_loss = actor_loss + (-rew_acc[i + 1, done_env_ids]).sum()
            else:
                actor_loss = actor_loss + (-rew_acc[i + 1, :]).sum()

            gamma = gamma * self.gamma
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(
                        self.episode_discounted_loss[done_env_ids])
                    self.episode_length_meter.update(
                        self.episode_length[done_env_ids].float())
                    for d in done_env_ids:
                        self.episode_loss_his.append(self.episode_loss[d].item())
                        self.episode_discounted_loss_his.append(
                            self.episode_discounted_loss[d].item())
                        self.episode_length_his.append(self.episode_length[d].item())
                        self.episode_loss[d] = 0.0
                        self.episode_discounted_loss[d] = 0.0
                        self.episode_length[d] = 0
                        self.episode_gamma[d] = 1.0

        actor_loss = actor_loss / (self.steps_num * self.num_envs)
        self.actor_loss = actor_loss.detach().cpu().item()
        self.step_count += self.steps_num * self.num_envs
        return actor_loss

    @torch.no_grad()
    def evaluate_policy(self, num_games: int, deterministic: bool = False):
        episode_length_his: list[int] = []
        episode_loss_his: list[float] = []
        episode_discounted_loss_his: list[float] = []
        episode_loss = torch.zeros(self.num_envs, dtype=torch.float32,
                                   device=self.device)
        episode_length = torch.zeros(self.num_envs, dtype=torch.long,
                                     device=self.device)
        episode_gamma = torch.ones(self.num_envs, dtype=torch.float32,
                                   device=self.device)
        episode_discounted_loss = torch.zeros(self.num_envs,
                                              dtype=torch.float32,
                                              device=self.device)
        obs = self.env.reset()
        games = 0
        while games < num_games:
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)
            actions = self.actor(obs, deterministic=deterministic)
            obs, rew, done, _ = self.env.step(torch.tanh(actions))
            episode_length += 1
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for d in done_env_ids:
                    episode_loss_his.append(episode_loss[d].item())
                    episode_discounted_loss_his.append(
                        episode_discounted_loss[d].item())
                    episode_length_his.append(episode_length[d].item())
                    episode_loss[d] = 0.0
                    episode_discounted_loss[d] = 0.0
                    episode_length[d] = 0
                    episode_gamma[d] = 1.0
                    games += 1
        return (float(np.mean(episode_loss_his)),
                float(np.mean(episode_discounted_loss_his)),
                float(np.mean(episode_length_his)))

    def initialize_env(self) -> None:
        self.env.clear_grad()
        self.env.reset()

    @torch.no_grad()
    def run(self, num_games: int) -> None:
        m, md, ml = self.evaluate_policy(
            num_games=num_games,
            deterministic=not self.stochastic_evaluation)
        print_info(f"mean episode loss = {m}, mean discounted loss = {md}, "
                   f"mean episode length = {ml}")

    def train(self) -> None:
        self.start_time = time.time()
        for n in ("algorithm", "compute actor loss", "forward simulation",
                  "backward simulation", "actor training"):
            self.time_report.add_timer(n)
        self.time_report.start_timer("algorithm")
        self.initialize_env()
        self.episode_loss = torch.zeros(self.num_envs, dtype=torch.float32,
                                        device=self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs,
                                                   dtype=torch.float32,
                                                   device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long,
                                        device=self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype=torch.float32,
                                        device=self.device)

        def actor_closure():
            self.actor_optimizer.zero_grad()
            self.time_report.start_timer("compute actor loss")
            self.time_report.start_timer("forward simulation")
            loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")
            self.time_report.start_timer("backward simulation")
            loss.backward()
            self.time_report.end_timer("backward simulation")
            with torch.no_grad():
                self.grad_norm_before_clip = grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm_max)
                self.grad_norm_after_clip = grad_norm(self.actor.parameters())
            self.time_report.end_timer("compute actor loss")
            return loss

        for epoch in range(self.max_epochs):
            t0 = time.time()
            if self.lr_schedule == "linear":
                lr = (1e-5 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for g in self.actor_optimizer.param_groups:
                    g["lr"] = lr
            else:
                lr = self.actor_lr
            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure)
            self.time_report.end_timer("actor training")
            self.iter_count += 1
            t1 = time.time()
            elapsed = time.time() - self.start_time

            self.writer.add_scalar("lr/iter", lr, self.iter_count)
            self.writer.add_scalar("actor_loss/step", self.actor_loss, self.step_count)
            self.writer.add_scalar("actor_loss/iter", self.actor_loss, self.iter_count)

            if len(self.episode_loss_his) > 0:
                mean_ep_len = float(self.episode_length_meter.get_mean())
                mean_pl = float(self.episode_loss_meter.get_mean())
                mean_pdl = float(self.episode_discounted_loss_meter.get_mean())
                if mean_pl < self.best_policy_loss:
                    print_info(f"save best policy with loss {mean_pl:.2f}")
                    self.save()
                    self.best_policy_loss = mean_pl
                self.writer.add_scalar("policy_loss/step", mean_pl, self.step_count)
                self.writer.add_scalar("policy_loss/iter", mean_pl, self.iter_count)
                self.writer.add_scalar("rewards/step", -mean_pl, self.step_count)
                if self.step_metrics_hook is not None:
                    self.step_metrics_hook(self.step_count, mean_pl, elapsed)
            else:
                mean_pl = float("inf")
                mean_pdl = float("inf")
                mean_ep_len = 0.0

            fps = self.steps_num * self.num_envs / max(t1 - t0, 1e-6)
            print(f"iter {self.iter_count}: ep loss {mean_pl:.4f}, "
                  f"ep discounted loss {mean_pdl:.4f}, "
                  f"ep len {mean_ep_len:.1f}, fps total {fps:.2f}, "
                  f"grad pre {float(self.grad_norm_before_clip):.2f}, "
                  f"post {float(self.grad_norm_after_clip):.2f}")
            self.writer.flush()
            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(self.name + f"policy_iter{self.iter_count}_reward{-mean_pl:.3f}")

        self.time_report.end_timer("algorithm")
        self.time_report.report()
        self.save("final_policy")
        self.episode_loss_his = np.array(self.episode_loss_his)
        np.save(os.path.join(self.log_dir, "episode_loss_his.npy"),
                self.episode_loss_his)
        self.run(self.num_envs)
        self.close()

    def play(self, cfg: dict) -> None:
        self.load(cfg["params"]["general"]["checkpoint"])
        self.run(cfg["params"]["config"]["player"]["games_num"])

    def save(self, filename: str | None = None) -> None:
        if filename is None:
            filename = "best_policy"
        torch.save([self.actor, self.obs_rms],
                   os.path.join(self.log_dir, f"{filename}.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=False)
        self.actor = ckpt[0].to(self.device)
        self.obs_rms = ckpt[1].to(self.device) if ckpt[1] is not None else None

    def close(self) -> None:
        self.writer.close()
