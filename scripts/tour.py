import time
import gymnasium as gym
import mujoco.viewer
import so101_nexus.mujoco

ENV_IDS = [
    "MuJoCoTouch-v1",
    "MuJoCoLookAt-v1",
    "MuJoCoMove-v1",
    "MuJoCoPickLift-v1",
    "MuJoCoPickAndPlace-v1",
    "MuJoCoStackCube-v1",
]

for env_id in ENV_IDS:
    print(f"\n--- {env_id} --- (~10 seconds)")
    env = gym.make(env_id, render_mode="human")
    obs, info = env.reset(seed=0)

    for _ in range(300):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()
        time.sleep(0.02)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    time.sleep(0.5)

print("\nTour complete.")