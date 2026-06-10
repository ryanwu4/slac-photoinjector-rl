from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch
import yaml
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from . import models
from .utils import (
    AverageMeter,
    RunningMeanStd,
    print_info,
    seeding,
    TimeReport,
)


class BPTT:
    def __init__(self, cfg, env_fn):
        #this initialization is mostly the same as the original paper
        #hand-copied by Ryan for understanding, but not modified substantively
        # we take the environmennt function as an argument, unlike in DiffRL
        seed = cfg["params"]["general"]["seed"]
        seeding(seed)

        config = cfg["params"]["config"]
        self.name = config.get("name", "bptt_emittance")

        #define the environment in the same way as the original paper does, though
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

        # env definitions same as in original paper
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length

        #from config
        self.device = cfg["params"]["general"]["device"]
        self.gamma = config.get("gamma", 0.99)
        self.steps_num = config.get("steps_num", self.max_episode_length)
        self.max_epochs = config.get("max_epochs", 1000)
        self.actor_learning_rate = float(config.get("actor_learning_rate", 1e-4))
        self.lr_schedule = config.get("lr_schedule", "linear")

        #object for tracking running observation stats for observations 
        self.observations_running_mean_std = None
        if config.get("obs_rms", False):
            self.observations_running_mean_std = RunningMeanStd(shape=self.num_obs, device=self.device)

        self.reward_scale = config.get("rew_scale", 1.0)
        self.truncate_grad = config.get("truncate_grads", False)
        self.grad_norm_max = config.get("grad_norm", 0.5)
        self.step_metrics_hook = None # set by driver for logging / eval callback, added by claude
        
        train = cfg['params']['general']['train']
        if train:
            #logging code by claude
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
        else: #evaluating
          self.stochastic_evaluation = not cfg['params']['config']['player'].get('deterministic', False)
          self.steps_num = self.env.episode_length

        #actor is stochastic MLP in these experiments
        self.actor = models.ActorStochasticMLP(
            obs_dim=self.num_obs,
            action_dim=self.num_actions,
            cfg_network=cfg["params"]["network"],
            device=self.device
        )

        # swapping these betas was done in the original paper
        #might need an ablation to check if this was necesasry really
        adam_betas = config.get("betas", [0.9, 0.999])
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_learning_rate,
            betas=adam_betas
        )

        #logging and loss, exactly the same as in DiffRL
        self.iter_count = 0
        self.step_count = 0
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int)
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        #time tracking for training algo parts
        self.time_report = TimeReport()


    def compute_actor_loss(self, deterministic = False):
        obs = self.env.initialize_trajectory()

        #initialize reward accumulation and discount value
        #rew_acc holds the accumulated reward 
        rew_acc = torch.zeros((self.steps_num+1, self.num_envs), dtype=torch.float32, device=self.device)
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

        actor_loss = torch.zeros(1, dtype=torch.float32, device=self.device)

        #added by claude from original paper, with running mean std
        if self.observations_running_mean_std is not None:
            with torch.no_grad():
                self.observations_running_mean_std.update(obs) #update first on initial observation
            obs = self.observations_running_mean_std.normalize(obs)

        #step through trajectory
        for i in range(self.steps_num):
            actions = self.actor(obs, deterministic=deterministic)
            obs, rew, done, info = self.env.step(torch.tanh(actions)) #tanh to bound like in original paper
            with torch.no_grad():   
                raw_rew = rew.clone()
            rew = rew *self.reward_scale

            #update and normalize again
            if self.observations_running_mean_std is not None:
                with torch.no_grad():
                    self.observations_running_mean_std.update(obs)
                obs = self.observations_running_mean_std.normalize(obs)

            self.episode_length += 1

            #accumulate the reward
            rew_acc[i+1] = rew_acc[i] + gamma * rew
            gamma = gamma * self.gamma #discount the reward for the next step

            terminal_envs = done.nonzero(as_tuple = False).squeeze(-1)
            if i<self.steps_num-1: #not the last step
                #add loss for the terminated envs
                actor_loss = actor_loss + (-rew_acc[i+1, terminal_envs]).sum()
            else: #last step, add loss for all envs
                actor_loss = actor_loss + (-rew_acc[i+1]).sum()

            #reset done envs
            rew_acc[i+1, terminal_envs] = 0.0
            gamma[terminal_envs] = 1.0

            #episode loss for evaluation, done by Claude like in diffRL
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(terminal_envs) > 0:
                    self.episode_loss_meter.update(self.episode_loss[terminal_envs])
                    self.episode_discounted_loss_meter.update(
                        self.episode_discounted_loss[terminal_envs])
                    self.episode_length_meter.update(
                        self.episode_length[terminal_envs].float())
                    for d in terminal_envs:
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
    def evaluate_policy(self, num_games, deterministic = False):
        num_episodes = num_games # just a nomenclature change since diffRL assumes a game is an episode

        obs = self.env.reset()
        num_envs = self.num_envs
        episode_loss = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        episode_discounted_loss = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        episode_gamma = torch.ones(num_envs, dtype=torch.float32, device=self.device)
        episode_length = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_length_his = []

        episodes_completed = 0
        while episodes_completed < num_episodes:
            if self.observations_running_mean_std is not None:
                #we don't recalibrate the rms during evaluation
                obs = self.observations_running_mean_std.normalize(obs)

            #take an action
            actions = self.actor(obs, deterministic=deterministic)
            obs, rew, done, info = self.env.step(torch.tanh(actions))
            episode_length += 1
            episode_loss -= rew #tracking total loss is useful here for emittance eval
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma

            terminal_envs = done.nonzero(as_tuple=False).squeeze(-1)
            if len(terminal_envs) > 0:
                for terminal_env in terminal_envs:
                    episode_loss_his.append(episode_loss[terminal_env].item())
                    episode_discounted_loss_his.append(episode_discounted_loss[terminal_env].item())
                    episode_length_his.append(episode_length[terminal_env].item())
                    episode_loss[terminal_env] = 0.0
                    episode_discounted_loss[terminal_env] = 0.0
                    episode_length[terminal_env] = 0
                    episode_gamma[terminal_env] = 1.0
                    episodes_completed += 1

        #mean statistics are expected
        mean_loss = np.mean(episode_loss_his)
        mean_discounted_loss = np.mean(episode_discounted_loss_his)
        mean_length = np.mean(episode_length_his)
        return mean_loss, mean_discounted_loss, mean_length


    def initialize_env(self):
        self.env.clear_grad()
        self.env.reset()

    #below training harnesses mostly written by claude, and similar to DiffRL
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

        #this is the important part!
        def actor_closure():
            #zero gradient for the optimizer first
            self.actor_optimizer.zero_grad()
            loss = self.compute_actor_loss()
            #backprop through the entire rollout
            loss.backward()
            with torch.no_grad():
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm_max)
            return loss

        for epoch in range(self.max_epochs):
            t0 = time.time()
            if self.lr_schedule == "linear":
                lr = (1e-5 - self.actor_learning_rate) * float(epoch / self.max_epochs) + self.actor_learning_rate
                for g in self.actor_optimizer.param_groups:
                    g["lr"] = lr
            else:
                lr = self.actor_learning_rate
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
                  f"ep len {mean_ep_len:.1f}, fps total {fps:.2f}, ")
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
        torch.save([self.actor, self.observations_running_mean_std],
                   os.path.join(self.log_dir, f"{filename}.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=False)
        self.actor = ckpt[0].to(self.device)
        self.observations_running_mean_std = ckpt[1].to(self.device) if ckpt[1] is not None else None

    def close(self) -> None:
        self.writer.close()
