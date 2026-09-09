"""
Run a trained ACT policy in its MuJoCo environment.

Usage:
    python eval_policy.py --inspect
        Show what keys the policy expects vs what the env provides.
        RUN THIS FIRST. Key mismatches are the #1 cause of a policy
        that loads fine but does nothing sensible.

    python eval_policy.py --watch
        Live 3D viewer, one episode at a time.

    python eval_policy.py --video
        Render episodes to outputs/eval/*.mp4

    python eval_policy.py --measure --episodes 20
        No rendering. Reports success rate.
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import mujoco.viewer
import so101_nexus.mujoco
from so101_nexus.config import PickConfig
from so101_nexus.observations import (
    JointPositions, JointVelocities, EndEffectorPose,
    GraspState, GazeState, ObjectPose, ObjectOffset,
    WristCamera, OverheadCamera,
)

CHECKPOINT = "outputs/act_picklift/checkpoints/020000/pretrained_model"
ENV_ID = "MuJoCoPickLift-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_policy():
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(CHECKPOINT)
    policy.to(DEVICE)
    policy.eval()
    return policy


def build_batch(obs, policy, env=None):
    """Map the env observation dict onto the keys the policy was trained with."""
    batch = {}
    expected = list(policy.config.input_features.keys())

    if "observation.environment_state" in expected and env is not None:
        priv = env.unwrapped._privileged_state
        if priv is None:
            raise RuntimeError("privileged_state is None — check obs_mode and cameras")
        obs = dict(obs)
        obs["state"] = np.rad2deg(np.asarray(obs["state"], dtype=np.float32))
        obs["environment_state"] = np.asarray(priv)

    for key in expected:
        if key in obs:
            value = obs[key]
        elif key.replace("observation.", "") in obs:
            value = obs[key.replace("observation.", "")]
        else:
            # Fall back: match on the last path segment.
            tail = key.split(".")[-1]
            match = next((k for k in obs if tail in k), None)
            if match is None:
                raise KeyError(
                    f"Policy expects '{key}' but the env gave {list(obs.keys())}.\n"
                    f"Run with --inspect and fix the mapping in build_batch()."
                )
            value = obs[match]

        t = torch.as_tensor(np.asarray(value))
        if t.ndim == 3 and t.shape[-1] in (1, 3):     # HWC image -> CHW
            t = t.permute(2, 0, 1)
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        batch[key] = t.unsqueeze(0).to(DEVICE).float()   # add batch dim

    return batch


def inspect():
    policy = load_policy()
    cfg = PickConfig(
    obs_mode="visual",
    observations=[
        JointPositions(),
        JointVelocities(),
        EndEffectorPose(),
        GraspState(),
        ObjectPose(),
        ObjectOffset(),
        WristCamera(),
        OverheadCamera(),
    ],
)
    env = gym.make(ENV_ID, config=cfg, render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    print("=== POLICY EXPECTS ===")
    for k, v in policy.config.input_features.items():
        print(f"  {k}: {v}")
    print("\n=== POLICY OUTPUTS ===")
    for k, v in policy.config.output_features.items():
        print(f"  {k}: {v}")

    print("\n=== ENV PROVIDES ===")
    if isinstance(obs, dict):
        for k, v in obs.items():
            print(f"  {k}: shape={np.asarray(v).shape} dtype={np.asarray(v).dtype}")
    else:
        print(f"  (not a dict) shape={np.asarray(obs).shape}")

    print("\n=== MAPPING TEST ===")
    try:
        batch = build_batch(obs, policy, env)
        for k, v in batch.items():
            print(f"  {k} -> {tuple(v.shape)} {v.dtype}")
        with torch.no_grad():
            action = policy.select_action(batch)
        print(f"\n  action out: {tuple(action.shape)}")
        print(f"  env expects: {env.action_space.shape}")
        print("\n  MAPPING OK" if action.shape[-1] == env.action_space.shape[0]
              else "\n  SHAPE MISMATCH - check output_features")
    except Exception as e:
        print(f"  FAILED: {e}")

    env.close()


def rollout(render_mode, episodes, save_video=False):
    import time
    policy = load_policy()
    cfg = PickConfig(
    obs_mode="visual",
    observations=[
        JointPositions(),
        JointVelocities(),
        EndEffectorPose(),
        GraspState(),
        ObjectPose(),
        ObjectOffset(),
        WristCamera(),
        OverheadCamera(),
    ],
)
    env = gym.make(ENV_ID, config=cfg, render_mode=render_mode)

    successes = 0
    all_rewards = []

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        if hasattr(policy, "reset"):
            policy.reset()          # clears the action-chunk queue between episodes

        frames = []
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 500:
            batch = build_batch(obs, policy, env)
            with torch.no_grad():
                action = policy.select_action(batch)
            action = action.squeeze(0).cpu().numpy()
            action = np.deg2rad(action)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            steps += 1

            if render_mode == "human":
                env.render()
                time.sleep(0.01)
            elif save_video:
                frames.append(env.render())

        success = bool(info.get("is_success", terminated))
        successes += int(success)
        all_rewards.append(total_reward)
        print(f"  episode {ep:3d}: steps={steps:4d} reward={total_reward:8.2f} "
              f"success={success}")

        if save_video and frames:
            import imageio
            os.makedirs("outputs/eval", exist_ok=True)
            path = f"outputs/eval/{ENV_ID}_ep{ep:03d}.mp4"
            imageio.mimsave(path, frames, fps=30)
            print(f"    saved {path}")

    env.close()

    print(f"\n=== RESULTS over {episodes} episodes ===")
    print(f"  success rate : {successes}/{episodes} = {100*successes/episodes:.1f}%")
    print(f"  mean reward  : {np.mean(all_rewards):.2f}")
    print(f"  std reward   : {np.std(all_rewards):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--measure", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}\n")

    if args.inspect:
        inspect()
    elif args.watch:
        rollout("human", args.episodes)
    elif args.video:
        rollout("rgb_array", args.episodes, save_video=True)
    elif args.measure:
        rollout("rgb_array", args.episodes)
    else:
        print("Pick one: --inspect | --watch | --video | --measure")