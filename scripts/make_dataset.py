"""
Generate a LeRobot dataset from the scripted pick-and-place controller.

Runs many seeds, keeps only successful episodes, writes them in
LeRobot v3 format ready for lerobot-train.

Records ONLY what a real robot could observe:
    observation.state            6 joint positions (radians)
    observation.images.wrist     480x640 RGB
    observation.images.overhead  480x640 RGB
    action                       6 joint targets (radians)

environment_state is deliberately excluded. That privileged cube pose
is what made the public checkpoint impossible to transfer to hardware.

Usage:
    python scripts\\make_dataset.py --check
        Verify the LeRobot dataset API matches this script. Run first.

    python scripts\\make_dataset.py --generate --episodes 200
        Run 200 seeds, keep the successes, write the dataset.

    python scripts\\make_dataset.py --verify
        Inspect what was written.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
import so101_nexus.mujoco

# reuse the controller you already tuned
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripted_pick import CFG, run_episode          # noqa: E402

REPO_ID = "cyrildev1/so101_pickplace_sim"
ROOT = Path(__file__).resolve().parent.parent / "data" / "so101_pickplace_sim"
TASK = "pick up the cube and place it to the side"

FPS = 15            # after subsampling
SUBSAMPLE = 2       # keep every Nth frame (env runs at 30 Hz)

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                  "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.overhead": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                  "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
    },
}


def import_dataset_class():
    """The import path has moved between LeRobot versions."""
    for path in ("lerobot.datasets.lerobot_dataset",
                 "lerobot.common.datasets.lerobot_dataset"):
        try:
            mod = __import__(path, fromlist=["LeRobotDataset"])
            return mod.LeRobotDataset
        except ImportError:
            continue
    sys.exit("Could not find LeRobotDataset. Run:\n"
             "  python -c \"import lerobot, pkgutil; "
             "print([m.name for m in pkgutil.iter_modules(lerobot.__path__)])\"")


def check():
    import inspect
    LeRobotDataset = import_dataset_class()
    print("LeRobotDataset found at:", LeRobotDataset.__module__)
    print("\n--- create() signature ---")
    print(inspect.signature(LeRobotDataset.create))
    print("\n--- add_frame() signature ---")
    print(inspect.signature(LeRobotDataset.add_frame))
    print("\n--- save_episode() signature ---")
    print(inspect.signature(LeRobotDataset.save_episode))
    print("\nIf these differ from what make_dataset.py calls, adjust before generating.")


def generate(n_episodes, keep_max):
    LeRobotDataset = import_dataset_class()

    if ROOT.exists():
        ans = input(f"{ROOT} exists. Delete and regenerate? [y/N] ")
        if ans.lower() != "y":
            sys.exit("Aborted.")
        shutil.rmtree(ROOT)

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=ROOT,
        robot_type="so101",
        use_videos=True,
    )

    env = gym.make("MuJoCoPickLift-v1", config=CFG,
                   render_mode="rgb_array", max_episode_steps=4000)

    kept = 0
    attempted = 0

    for seed in range(n_episodes):
        attempted += 1
        ok, _frames, transitions = run_episode(env, seed, collect=True)

        if not ok:
            print(f"  seed {seed:3d}: FAIL   (kept {kept})")
            continue

        frames = transitions[::SUBSAMPLE]
        for tr in frames:
            ds.add_frame({
                "observation.state": tr["state"].astype(np.float32),
                "observation.images.wrist": tr["wrist"],
                "observation.images.overhead": tr["overhead"],
                "action": tr["action"].astype(np.float32),
                "task": TASK,
            })
        ds.save_episode()
        kept += 1
        print(f"  seed {seed:3d}: OK     {len(frames):4d} frames  (kept {kept})")

        if keep_max and kept >= keep_max:
            print(f"\nReached target of {keep_max} episodes.")
            break

    env.close()

    print(f"\n=== {kept} episodes kept from {attempted} attempts "
          f"({100*kept/max(attempted,1):.1f}%) ===")
    print(f"Written to: {ROOT}")
    print("\nNext: python scripts\\make_dataset.py --verify")


def verify():
    LeRobotDataset = import_dataset_class()
    ds = LeRobotDataset(REPO_ID, root=ROOT)

    print(f"episodes : {ds.num_episodes}")
    print(f"frames   : {ds.num_frames}")
    print(f"fps      : {ds.fps}")
    print(f"\nfeatures : {list(ds.features.keys())}")

    f = ds[0]
    print("\n--- first frame ---")
    for k, v in f.items():
        if hasattr(v, "shape"):
            arr = np.asarray(v)
            extra = ""
            if arr.dtype.kind == "f" and arr.size < 20:
                extra = f"  values={np.round(arr, 3)}"
            print(f"  {k:32s} {str(tuple(arr.shape)):20s} {arr.dtype}{extra}")
        else:
            print(f"  {k:32s} {v}")

    # sanity: state should be radians, roughly within joint limits
    st = np.asarray(f["observation.state"])
    if np.abs(st).max() > 6.5:
        print("\n  WARNING: state values look too large for radians.")
    else:
        print("\n  state range looks like radians - good")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--episodes", type=int, default=200,
                   help="how many seeds to attempt")
    p.add_argument("--keep", type=int, default=0,
                   help="stop after this many successes (0 = no limit)")
    a = p.parse_args()

    if a.check:
        check()
    elif a.generate:
        generate(a.episodes, a.keep)
    elif a.verify:
        verify()
    else:
        print("Pick one: --check | --generate | --verify")