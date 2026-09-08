import gymnasium as gym
import mujoco.viewer
import so101_nexus.mujoco
import imageio
import numpy as np

env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")   # NOT human
obs, info = env.reset(seed=0)

n = env.action_space.shape[0]
frames = []

for step in range(300):
    action = np.zeros(n, dtype=np.float32)
    joint = (step // 50) % n
    phase = (step % 50) / 50.0
    action[joint] = 0.6 * np.sin(phase * 2 * np.pi)
    action = np.clip(action, env.action_space.low, env.action_space.high)

    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        obs, info = env.reset()

env.close()

imageio.mimsave("rollout.mp4", frames, fps=30)
print(f"Saved rollout.mp4 ({len(frames)} frames)")