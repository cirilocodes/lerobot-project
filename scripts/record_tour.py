import gymnasium as gym
import mujoco.viewer
import so101_nexus.mujoco
import imageio
import os

ENV_IDS = [
    "MuJoCoTouch-v1",
    "MuJoCoLookAt-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
    "MuJoCoStackCube-v1",
]

OUT_DIR = "outputs/tour"
STEPS = 300
FPS = 30

os.makedirs(OUT_DIR, exist_ok=True)

for env_id in ENV_IDS:
    print(f"\n--- {env_id} ---")
    env = gym.make(env_id, render_mode="rgb_array")   # NOT human
    obs, info = env.reset(seed=0)

    frames = []
    for _ in range(STEPS):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    path = os.path.join(OUT_DIR, f"{env_id}.mp4")
    imageio.mimsave(path, frames, fps=FPS)
    print(f"  saved {path}  ({len(frames)} frames)")

print(f"\nDone. Videos in {OUT_DIR}\\")