#!/usr/bin/env python3
"""
Minimal single-file educational implementation of "agentic RL" with GRPO in PyTorch.

No external dependencies beyond torch + Python stdlib.

What this script demonstrates:
- A tiny policy network controlling an agent over multi-step episodes (agentic behavior).
- Grouped sampling per task instance.
- GRPO-style relative advantages within each group.
- PPO-style clipped objective + optional KL-to-reference penalty.

This is intentionally compact and pedagogical, not production RL infrastructure.
"""

import copy
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Toy "agentic" environment
# -----------------------------
class LineWorldTask:
    """
    1D line world. Agent starts at 0, target is in [0, horizon].

    Actions:
      0 -> move +1
      1 -> move -1 (floored at 0)
      2 -> STOP

    Episode ends when STOP or max_steps.

    Reward:
      +1.0 if STOP exactly on target
      -0.05 * distance_to_target
      -0.01 * steps_used   (encourage shorter plans)
    """

    def __init__(self, target: int, horizon: int = 8, max_steps: int = 8):
        self.target = target
        self.horizon = horizon
        self.max_steps = max_steps
        self.pos = 0
        self.steps = 0
        self.done = False

    def obs(self) -> torch.Tensor:
        # Observation: [pos_norm, target_norm, delta_norm, step_norm]
        pos = self.pos / self.horizon
        tgt = self.target / self.horizon
        delta = (self.target - self.pos) / self.horizon
        step = self.steps / self.max_steps
        return torch.tensor([pos, tgt, delta, step], dtype=torch.float32)

    def step(self, action: int) -> Tuple[torch.Tensor, bool]:
        if self.done:
            return self.obs(), True

        if action == 0:
            self.pos = min(self.horizon, self.pos + 1)
        elif action == 1:
            self.pos = max(0, self.pos - 1)
        elif action == 2:
            self.done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self.obs(), self.done

    def final_reward(self) -> float:
        dist = abs(self.pos - self.target)
        hit = 1.0 if (self.done and self.pos == self.target) else 0.0
        return hit - 0.05 * dist - 0.01 * self.steps


# -----------------------------
# Policy
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = 4, hidden: int = 64, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class Trajectory:
    obs: List[torch.Tensor]
    actions: List[int]
    old_logprobs: List[float]
    ref_logprobs: List[float]
    reward: float


def sample_trajectory(policy: PolicyNet, ref_policy: PolicyNet, target: int, max_steps: int, device: torch.device) -> Trajectory:
    env = LineWorldTask(target=target, max_steps=max_steps)
    obs_list, act_list, old_lp_list, ref_lp_list = [], [], [], []

    while True:
        obs = env.obs().to(device)
        logits = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        with torch.no_grad():
            ref_logits = ref_policy(obs)
            ref_dist = torch.distributions.Categorical(logits=ref_logits)
            ref_logp = ref_dist.log_prob(action)

        obs_list.append(obs.detach().cpu())
        act_list.append(int(action.item()))
        old_lp_list.append(float(dist.log_prob(action).detach().cpu().item()))
        ref_lp_list.append(float(ref_logp.detach().cpu().item()))

        _, done = env.step(int(action.item()))
        if done:
            break

    reward = env.final_reward()
    return Trajectory(obs_list, act_list, old_lp_list, ref_lp_list, reward)


def group_relative_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """GRPO core: normalize rewards within each group (same task prompt/instance)."""
    r = torch.tensor(rewards, dtype=torch.float32)
    mean = r.mean()
    std = r.std(unbiased=False)
    adv = (r - mean) / (std + eps)
    return adv.tolist()


def build_batch(grouped_trajs: List[List[Trajectory]], grouped_advs: List[List[float]], device: torch.device):
    """
    Flatten trajectories into token/action-level training rows.
    Each action in a trajectory gets the trajectory-level relative advantage.
    """
    obs_t, act_t, old_lp_t, ref_lp_t, adv_t = [], [], [], [], []

    for trajs, advs in zip(grouped_trajs, grouped_advs):
        for traj, a in zip(trajs, advs):
            for o, act, old_lp, ref_lp in zip(traj.obs, traj.actions, traj.old_logprobs, traj.ref_logprobs):
                obs_t.append(o)
                act_t.append(act)
                old_lp_t.append(old_lp)
                ref_lp_t.append(ref_lp)
                adv_t.append(a)

    return (
        torch.stack(obs_t).to(device),
        torch.tensor(act_t, dtype=torch.long, device=device),
        torch.tensor(old_lp_t, dtype=torch.float32, device=device),
        torch.tensor(ref_lp_t, dtype=torch.float32, device=device),
        torch.tensor(adv_t, dtype=torch.float32, device=device),
    )


def grpo_step(
    policy: PolicyNet,
    optimizer: torch.optim.Optimizer,
    grouped_trajs: List[List[Trajectory]],
    clip_eps: float,
    kl_beta: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    grouped_rewards = [[t.reward for t in g] for g in grouped_trajs]
    grouped_advs = [group_relative_advantages(r) for r in grouped_rewards]

    obs, actions, old_logp, ref_logp, adv = build_batch(grouped_trajs, grouped_advs, device)

    logits = policy(obs)
    dist = torch.distributions.Categorical(logits=logits)
    new_logp = dist.log_prob(actions)

    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    ppo_obj = torch.minimum(unclipped, clipped)

    # Sampled KL approximation against frozen reference policy
    approx_kl = (new_logp - ref_logp)

    # Maximize objective => minimize negative objective
    loss = -(ppo_obj - kl_beta * approx_kl).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return (
        float(loss.item()),
        float(ppo_obj.mean().detach().cpu().item()),
        float(approx_kl.mean().detach().cpu().item()),
    )


def evaluate(policy: PolicyNet, n_tasks: int, max_steps: int, device: torch.device) -> float:
    policy.eval()
    with torch.no_grad():
        total = 0.0
        for _ in range(n_tasks):
            target = random.randint(0, max_steps)
            traj = sample_trajectory(policy, policy, target, max_steps, device)
            total += traj.reward
    policy.train()
    return total / n_tasks


def main():
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparams (small, educational) ---
    updates = 300
    tasks_per_update = 32      # number of different prompts/tasks per update
    group_size = 8             # trajectories sampled per task (GRPO grouping)
    max_steps = 8

    lr = 3e-4
    clip_eps = 0.2
    kl_beta = 0.02

    policy = PolicyNet().to(device)
    ref_policy = copy.deepcopy(policy).to(device).eval()  # fixed reference
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    print("Starting training...")
    print(f"device={device}, updates={updates}, tasks/update={tasks_per_update}, group_size={group_size}")

    for step in range(1, updates + 1):
        grouped_trajs = []

        # Collect grouped trajectories (same target within each group)
        for _ in range(tasks_per_update):
            target = random.randint(0, max_steps)
            group = [sample_trajectory(policy, ref_policy, target, max_steps, device) for _ in range(group_size)]
            grouped_trajs.append(group)

        loss, ppo_obj, kl = grpo_step(
            policy=policy,
            optimizer=optimizer,
            grouped_trajs=grouped_trajs,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
            device=device,
        )

        if step % 25 == 0 or step == 1:
            avg_group_reward = sum(t.reward for g in grouped_trajs for t in g) / (tasks_per_update * group_size)
            eval_reward = evaluate(policy, n_tasks=128, max_steps=max_steps, device=device)
            print(
                f"step={step:03d} "
                f"train_reward={avg_group_reward:+.3f} "
                f"eval_reward={eval_reward:+.3f} "
                f"loss={loss:+.4f} "
                f"ppo_obj={ppo_obj:+.4f} "
                f"kl={kl:+.4f}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
