"""
Short-Horizon Actor-Critic (SHAC), vendored from NVlabs/DiffRL.

Differences from upstream `DiffRL/algorithms/shac.py`:
  1. Removed `import dflex` and `import envs` (we drive an MLP surrogate, not
     a dflex physics sim).
  2. Replaced `env_fn = getattr(envs, cfg[...])` with a caller-supplied
     `env_fn` argument so the consumer wires the env directly. The cfg still
     carries the env name string for logging.
  3. Swapped `tensorboardX.SummaryWriter` for `torch.utils.tensorboard.SummaryWriter`
     (newer torch ships the latter natively).
  4. Imports `actor`/`critic` classes from `.models` and utils from `.utils`.
  5. Added a `step_metrics_hook` callback so external runners can pump a
     learning-curve CSV without subclassing.

Algorithm code (compute_actor_loss, compute_target_values, training loop,
critic update, target-critic EMA) is unchanged.
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
    CriticDataset,
    RunningMeanStd,
    TimeReport,
    grad_norm,
    print_info,
    seeding,
)


class SHAC:
    def __init__(self, cfg: dict, env_fn: Callable):
        seeding(cfg["params"]["general"]["seed"])
        diff_env_cfg = cfg["params"]["diff_env"]
        self.env = env_fn(
            num_envs=cfg["params"]["config"]["num_actors"],
            device=cfg["params"]["general"]["device"],
            render=cfg["params"]["general"]["render"],
            seed=cfg["params"]["general"]["seed"],
            episode_length=diff_env_cfg.get("episode_length", 250),
            stochastic_init=diff_env_cfg.get("stochastic_env", True),
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
        self.critic_method = cfg["params"]["config"].get("critic_method",
                                                         "one-step")
        if self.critic_method == "td-lambda":
            self.lam = cfg["params"]["config"].get("lambda", 0.95)

        self.steps_num = cfg["params"]["config"]["steps_num"]
        self.max_epochs = cfg["params"]["config"]["max_epochs"]
        self.actor_lr = float(cfg["params"]["config"]["actor_learning_rate"])
        self.critic_lr = float(cfg["params"]["config"]["critic_learning_rate"])
        self.lr_schedule = cfg["params"]["config"].get("lr_schedule", "linear")
        self.target_critic_alpha = cfg["params"]["config"].get(
            "target_critic_alpha", 0.4)

        self.obs_rms: Optional[RunningMeanStd] = None
        if cfg["params"]["config"].get("obs_rms", False):
            self.obs_rms = RunningMeanStd(shape=(self.num_obs,),
                                          device=self.device)
        self.ret_rms: Optional[RunningMeanStd] = None
        if cfg["params"]["config"].get("ret_rms", False):
            self.ret_rms = RunningMeanStd(shape=(), device=self.device)

        self.rew_scale = cfg["params"]["config"].get("rew_scale", 1.0)
        self.critic_iterations = cfg["params"]["config"].get(
            "critic_iterations", 16)
        self.num_batch = cfg["params"]["config"].get("num_batch", 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg["params"]["config"].get("name", "Photoinjector")
        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]
        self.grad_norm_max = cfg["params"]["config"]["grad_norm"]
        self.step_metrics_hook: Optional[Callable[[int, float, float], None]] = None

        if cfg["params"]["general"]["train"]:
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok=True)
            save_cfg = copy.deepcopy(cfg)
            if "general" in save_cfg["params"]:
                deleted_keys = [k for k in save_cfg["params"]["general"]
                                if k in save_cfg["params"]["config"]]
                for k in deleted_keys:
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

        # actor / critic
        self.actor_name = cfg["params"]["network"].get("actor", "ActorStochasticMLP")
        self.critic_name = cfg["params"]["network"].get("critic", "CriticMLP")
        actor_fn = getattr(_models, self.actor_name)
        critic_fn = getattr(_models, self.critic_name)
        self.actor = actor_fn(self.num_obs, self.num_actions,
                              cfg["params"]["network"], device=self.device)
        self.critic = critic_fn(self.num_obs, cfg["params"]["network"],
                                device=self.device)
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.target_critic = copy.deepcopy(self.critic)

        if cfg["params"]["general"]["train"]:
            self.save("init_policy")

        betas = cfg["params"]["config"].get("betas", [0.7, 0.95])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                betas=betas, lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 betas=betas, lr=self.critic_lr)

        self.obs_buf = torch.zeros((self.steps_num, self.num_envs, self.num_obs),
                                   dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((self.steps_num, self.num_envs),
                                   dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((self.steps_num, self.num_envs),
                                     dtype=torch.float32, device=self.device)
        self.next_values = torch.zeros((self.steps_num, self.num_envs),
                                       dtype=torch.float32, device=self.device)
        self.target_values = torch.zeros((self.steps_num, self.num_envs),
                                         dtype=torch.float32, device=self.device)
        self.ret = torch.zeros((self.num_envs,), dtype=torch.float32,
                               device=self.device)

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
        self.value_loss = np.inf

        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        self.time_report = TimeReport()
        self.grad_norm_before_clip = torch.tensor(0.0)
        self.grad_norm_after_clip = torch.tensor(0.0)

    # ---- core actor-loss computation -----------------------------------

    def compute_actor_loss(self, deterministic: bool = False) -> torch.Tensor:
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs),
                              dtype=torch.float32, device=self.device)
        gamma = torch.ones(self.num_envs, dtype=torch.float32,
                           device=self.device)
        next_values = torch.zeros((self.steps_num + 1, self.num_envs),
                                  dtype=torch.float32, device=self.device)

        actor_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)
            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        obs = self.env.initialize_trajectory()
        if self.obs_rms is not None:
            with torch.no_grad():
                self.obs_rms.update(obs)
            obs = obs_rms.normalize(obs)

        for i in range(self.steps_num):
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            actions = self.actor(obs, deterministic=deterministic)
            obs, rew, done, extra_info = self.env.step(torch.tanh(actions))

            with torch.no_grad():
                raw_rew = rew.clone()
            rew = rew * self.rew_scale

            if self.obs_rms is not None:
                with torch.no_grad():
                    self.obs_rms.update(obs)
                obs = obs_rms.normalize(obs)

            if self.ret_rms is not None:
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)
                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            next_values[i + 1] = self.target_critic(obs).squeeze(-1)

            for id_ in done_env_ids:
                obr = extra_info["obs_before_reset"][id_]
                if (torch.isnan(obr).sum() > 0 or torch.isinf(obr).sum() > 0
                        or (torch.abs(obr) > 1e6).sum() > 0):
                    next_values[i + 1, id_] = 0.0
                elif self.episode_length[id_] < self.max_episode_length:
                    next_values[i + 1, id_] = 0.0
                else:
                    real_obs = (obs_rms.normalize(obr) if self.obs_rms is not None
                                else obr)
                    next_values[i + 1, id_] = self.target_critic(real_obs).squeeze(-1)

            if ((next_values[i + 1] > 1e6).sum() > 0
                    or (next_values[i + 1] < -1e6).sum() > 0):
                raise ValueError("next value error")

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                actor_loss = actor_loss + (
                    -rew_acc[i + 1, done_env_ids]
                    - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]
                ).sum()
            else:
                actor_loss = actor_loss + (
                    -rew_acc[i + 1, :]
                    - self.gamma * gamma * next_values[i + 1, :]
                ).sum()

            gamma = gamma * self.gamma
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                self.next_values[i] = next_values[i + 1].clone()

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
                    for done_env_id in done_env_ids:
                        if (self.episode_loss[done_env_id] > 1e6
                                or self.episode_loss[done_env_id] < -1e6):
                            raise ValueError("ep loss error")
                        self.episode_loss_his.append(
                            self.episode_loss[done_env_id].item())
                        self.episode_discounted_loss_his.append(
                            self.episode_discounted_loss[done_env_id].item())
                        self.episode_length_his.append(
                            self.episode_length[done_env_id].item())
                        self.episode_loss[done_env_id] = 0.0
                        self.episode_discounted_loss[done_env_id] = 0.0
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.0

        actor_loss = actor_loss / (self.steps_num * self.num_envs)
        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)
        self.actor_loss = actor_loss.detach().cpu().item()
        self.step_count += self.steps_num * self.num_envs
        return actor_loss

    # ---- eval ------------------------------------------------------------

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
        episode_discounted_loss = torch.zeros(self.num_envs, dtype=torch.float32,
                                              device=self.device)

        obs = self.env.reset()
        games_cnt = 0
        while games_cnt < num_games:
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
                    games_cnt += 1

        return (float(np.mean(episode_loss_his)),
                float(np.mean(episode_discounted_loss_his)),
                float(np.mean(episode_length_his)))

    @torch.no_grad()
    def compute_target_values(self) -> None:
        if self.critic_method == "one-step":
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == "td-lambda":
            Ai = torch.zeros(self.num_envs, dtype=torch.float32,
                             device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32,
                             device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32,
                             device=self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1.0 - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (
                    self.lam * self.gamma * Ai
                    + self.gamma * self.next_values[i]
                    + (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i])
                Bi = self.gamma * (
                    self.next_values[i] * self.done_mask[i]
                    + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError

    def compute_critic_loss(self, batch_sample):
        predicted = self.critic(batch_sample["obs"]).squeeze(-1)
        target = batch_sample["target_values"]
        return ((predicted - target) ** 2).mean()

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

    # ---- training loop --------------------------------------------------

    def train(self) -> None:
        self.start_time = time.time()
        for n in ("algorithm", "compute actor loss", "forward simulation",
                  "backward simulation", "prepare critic dataset",
                  "actor training", "critic training"):
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
            actor_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")
            self.time_report.start_timer("backward simulation")
            actor_loss.backward()
            self.time_report.end_timer("backward simulation")
            with torch.no_grad():
                self.grad_norm_before_clip = grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm_max)
                self.grad_norm_after_clip = grad_norm(self.actor.parameters())
                if (torch.isnan(self.grad_norm_before_clip)
                        or self.grad_norm_before_clip > 1e6):
                    raise ValueError("NaN gradient")
            self.time_report.end_timer("compute actor loss")
            return actor_loss

        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()
            if self.lr_schedule == "linear":
                lr = (1e-5 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for g in self.actor_optimizer.param_groups:
                    g["lr"] = lr
                critic_lr = (1e-5 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for g in self.critic_optimizer.param_groups:
                    g["lr"] = critic_lr
            else:
                lr = self.actor_lr

            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure)
            self.time_report.end_timer("actor training")

            # critic training
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(self.batch_size, self.obs_buf,
                                        self.target_values, drop_last=False)
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.0
            for _j in range(self.critic_iterations):
                total_critic_loss = 0.0
                batch_cnt = 0
                for i in range(len(dataset)):
                    bs = dataset[i]
                    self.critic_optimizer.zero_grad()
                    loss = self.compute_critic_loss(bs)
                    loss.backward()
                    for p in self.critic.parameters():
                        if p.grad is not None:
                            p.grad.nan_to_num_(0.0, 0.0, 0.0)
                    if self.truncate_grad:
                        clip_grad_norm_(self.critic.parameters(),
                                        self.grad_norm_max)
                    self.critic_optimizer.step()
                    total_critic_loss += loss
                    batch_cnt += 1
                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
            self.time_report.end_timer("critic training")

            self.iter_count += 1
            time_end_epoch = time.time()
            time_elapse = time.time() - self.start_time

            self.writer.add_scalar("lr/iter", lr, self.iter_count)
            self.writer.add_scalar("actor_loss/step", self.actor_loss,
                                   self.step_count)
            self.writer.add_scalar("actor_loss/iter", self.actor_loss,
                                   self.iter_count)
            self.writer.add_scalar("value_loss/step", self.value_loss,
                                   self.step_count)
            self.writer.add_scalar("value_loss/iter", self.value_loss,
                                   self.iter_count)

            if len(self.episode_loss_his) > 0:
                mean_ep_len = float(self.episode_length_meter.get_mean())
                mean_pl = float(self.episode_loss_meter.get_mean())
                mean_pdl = float(self.episode_discounted_loss_meter.get_mean())
                if mean_pl < self.best_policy_loss:
                    print_info(f"save best policy with loss {mean_pl:.2f}")
                    self.save()
                    self.best_policy_loss = mean_pl
                self.writer.add_scalar("policy_loss/step", mean_pl, self.step_count)
                self.writer.add_scalar("policy_loss/time", mean_pl, time_elapse)
                self.writer.add_scalar("policy_loss/iter", mean_pl, self.iter_count)
                self.writer.add_scalar("rewards/step", -mean_pl, self.step_count)
                self.writer.add_scalar("rewards/iter", -mean_pl, self.iter_count)
                self.writer.add_scalar("policy_discounted_loss/step",
                                       mean_pdl, self.step_count)
                self.writer.add_scalar("best_policy_loss/step",
                                       self.best_policy_loss, self.step_count)
                self.writer.add_scalar("episode_lengths/iter", mean_ep_len,
                                       self.iter_count)
                if self.step_metrics_hook is not None:
                    self.step_metrics_hook(self.step_count, mean_pl, time_elapse)
            else:
                mean_pl = float("inf")
                mean_pdl = float("inf")
                mean_ep_len = 0.0

            fps = self.steps_num * self.num_envs / max(time_end_epoch
                                                       - time_start_epoch, 1e-6)
            print(f"iter {self.iter_count}: ep loss {mean_pl:.4f}, "
                  f"ep discounted loss {mean_pdl:.4f}, "
                  f"ep len {mean_ep_len:.1f}, fps total {fps:.2f}, "
                  f"value loss {self.value_loss:.4f}, "
                  f"grad pre {float(self.grad_norm_before_clip):.2f}, "
                  f"post {float(self.grad_norm_after_clip):.2f}")
            self.writer.flush()

            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(self.name + f"policy_iter{self.iter_count}_reward{-mean_pl:.3f}")

            with torch.no_grad():
                a = self.target_critic_alpha
                for p, pt in zip(self.critic.parameters(),
                                 self.target_critic.parameters()):
                    pt.data.mul_(a)
                    pt.data.add_((1.0 - a) * p.data)

        self.time_report.end_timer("algorithm")
        self.time_report.report()
        self.save("final_policy")

        self.episode_loss_his = np.array(self.episode_loss_his)
        self.episode_discounted_loss_his = np.array(self.episode_discounted_loss_his)
        self.episode_length_his = np.array(self.episode_length_his)
        np.save(os.path.join(self.log_dir, "episode_loss_his.npy"),
                self.episode_loss_his)
        np.save(os.path.join(self.log_dir, "episode_discounted_loss_his.npy"),
                self.episode_discounted_loss_his)
        np.save(os.path.join(self.log_dir, "episode_length_his.npy"),
                self.episode_length_his)

        self.run(self.num_envs)
        self.close()

    def play(self, cfg: dict) -> None:
        self.load(cfg["params"]["general"]["checkpoint"])
        self.run(cfg["params"]["config"]["player"]["games_num"])

    def save(self, filename: str | None = None) -> None:
        if filename is None:
            filename = "best_policy"
        torch.save([self.actor, self.critic, self.target_critic,
                    self.obs_rms, self.ret_rms],
                   os.path.join(self.log_dir, f"{filename}.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=False)
        self.actor = ckpt[0].to(self.device)
        self.critic = ckpt[1].to(self.device)
        self.target_critic = ckpt[2].to(self.device)
        self.obs_rms = ckpt[3].to(self.device) if ckpt[3] is not None else None
        self.ret_rms = ckpt[4].to(self.device) if ckpt[4] is not None else None

    def close(self) -> None:
        self.writer.close()
