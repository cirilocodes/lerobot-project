import gymnasium as gym
import mujoco.viewer
import so101_nexus.mujoco
import numpy as np

env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")
obs, info = env.reset(seed=0)

print("=== OBSERVATION ===")
if isinstance(obs, dict):
    for k, v in obs.items():
        arr = np.asarray(v)
        print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
else:
    print("  array shape:", np.asarray(obs).shape)

print("\n=== ACTION ===")
print("  shape:", env.action_space.shape)
print("  low :", env.action_space.low)
print("  high:", env.action_space.high)

print("\n=== ONE STEP ===")
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("  reward:", reward)
print("  terminated:", terminated, " truncated:", truncated)
print("  info keys:", list(info.keys()))

env.close()