# test_debug.py
"""
Debug script to verify normalized->real action mapping and reward function.

What it checks:
 - deepracer-v0 env is registered (import deepracer_gym)
 - env.action_space is [-1,1] now (normalized)
 - For a small set of normalized actions, prints:
     * normalized action
     * simulator returned reward and info['reward_params'] (steering/speed/progress...)
     * a local mapped real steering/speed computed from configs/agent_params.json
     * difference between simulator steering/speed and local mapping
     * local reward (recomputed via configs/reward_function.reward_function) if importable
Notes:
 - Run from repo root, with the same conda/env as your training run.
 - If deepracer_gym isn't installed, the script will attempt to import from packages/deepracer_gym.
"""
import os
import sys
import json
import time
import numpy as np
import importlib

# Ensure repository root is in sys.path (adjust if you run from different cwd)
repo_root = os.path.abspath(os.getcwd())
sys.path.insert(0, repo_root)

# Try to import deepracer_gym (either installed or from local packages)
try:
    import deepracer_gym
    print("Imported deepracer_gym (installed).")
except Exception as e:
    local_pkg = os.path.join(repo_root, "packages", "deepracer_gym")
    if os.path.isdir(local_pkg):
        print("deepracer_gym not installed; adding local package path:", local_pkg)
        sys.path.insert(0, local_pkg)
        try:
            import deepracer_gym
            print("Imported deepracer_gym from local packages")
        except Exception as e2:
            print("Failed to import deepracer_gym locally:", e2)
            raise
    else:
        raise RuntimeError("deepracer_gym not installed and local package not found at packages/deepracer_gym")

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# Utility: load agent_params.json locally to read real action ranges
AGENT_PARAMS_PATH = os.path.join(repo_root, "configs", "agent_params.json")
agent_params = None
if os.path.exists(AGENT_PARAMS_PATH):
    try:
        with open(AGENT_PARAMS_PATH, "r") as fh:
            agent_params = json.load(fh)
            print("Loaded local configs/agent_params.json")
    except Exception as e:
        print("Failed to load configs/agent_params.json:", e)
else:
    print(f"configs/agent_params.json not found at {AGENT_PARAMS_PATH} - can't compute local mapping")

def denormalize(a_norm, low, high):
    """Map a_norm in [-1,1] to real range [low,high]."""
    scaled = (a_norm + 1.0) / 2.0  # [0,1]
    return float(low + scaled * (high - low))

# Try to import local reward function (optional)
reward_fn = None
try:
    reward_mod = importlib.import_module("configs.reward_function")
    reward_fn = getattr(reward_mod, "reward_function", None)
    if reward_fn is not None:
        print("Loaded configs.reward_function.reward_function")
    else:
        print("configs.reward_function found but missing reward_function")
except Exception as e:
    print("Could not import configs.reward_function:", e)

def print_section(msg):
    print("\n" + "="*8 + " " + msg + " " + "="*8)

def main():
    # create environment (do not use RecordEpisodeStatistics wrapper)
    # request rgb_array so render() returns array if implemented
    try:
        env = gym.make("deepracer-v0", render_mode="rgb_array")
    except TypeError:
        # some builds ignore render_mode param; fallback
        env = gym.make("deepracer-v0")
    env = FlattenObservation(env)  # flatten obs for convenience

    env.action_space.seed(0)
    env.observation_space.seed(0)

    print_section("ENV INFO")
    print("Gym version:", __import__("gymnasium").__version__)
    print("Action space:", env.action_space)
    try:
        print("Action space low:", env.action_space.low, "high:", env.action_space.high)
    except Exception:
        print("Action space (no low/high) ->", env.action_space)

    # Print agent_params local ranges (if loaded)
    if agent_params is not None:
        try:
            aspace = agent_params.get("action_space", {})
            sa = aspace.get("steering_angle", {})
            sp = aspace.get("speed", {})
            print("Local agent_params steering low/high:", sa.get("low"), sa.get("high"))
            print("Local agent_params speed   low/high:", sp.get("low"), sp.get("high"))
        except Exception as e:
            print("Error reading agent_params:", e)
    else:
        print("No local agent_params to compare mapping.")

    # Reset env once
    obs, _ = env.reset()
    time.sleep(0.2)

    # define test actions (normalized)
    tests = [
        np.array([-1.0, -1.0], dtype=np.float32),
        np.array([ 1.0,  1.0], dtype=np.float32),
        np.array([ 0.0,  0.0], dtype=np.float32),
        np.array([ 0.5, -0.5], dtype=np.float32),
        np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)], dtype=np.float32),
    ]

    print_section("TESTS")
    for i, a in enumerate(tests):
        # sanity assert: normalized action must be within env bounds
        low = getattr(env.action_space, "low", None)
        high = getattr(env.action_space, "high", None)
        if low is not None and high is not None:
            if not (np.all(a >= low - 1e-6) and np.all(a <= high + 1e-6)):
                print(f"WARNING: action {a} outside env.action_space bounds {low}/{high}")

        next_obs, reward_env, terminated, truncated, info = env.step(a)
        rp = info.get("reward_params", {})

        print(f"\n--- Test {i+1} ---")
        print("Normalized action:", a)
        print("Env returned reward:", reward_env)
        print("Info keys:", sorted(list(info.keys())))
        # print notable values
        for key in ["steering_angle", "speed", "progress", "distance_from_center", "is_crashed", "is_offtrack"]:
            if key in rp:
                print(f"  reward_params[{key}] = {rp.get(key)}")
        # show full reward_params (truncated)
        if rp:
            rp_items = list(rp.items())[:12]
            print("  reward_params sample:", rp_items, ("(...)" if len(rp_items) < len(rp) else ""))

        # compare with mapping computed from local agent_params (if available)
        if agent_params is not None:
            try:
                aspace = agent_params["action_space"]
                st = aspace["steering_angle"]
                sp = aspace["speed"]
                st_low, st_high = float(st["low"]), float(st["high"])
                sp_low, sp_high = float(sp["low"]), float(sp["high"])
                # mapping
                steering_mapped = denormalize(a[0], st_low, st_high)
                speed_mapped = denormalize(a[1], sp_low, sp_high)
                print("  locally mapped steering:", steering_mapped)
                print("  locally mapped speed:   ", speed_mapped)
                # if simulator returned them, print differences
                if "steering_angle" in rp:
                    try:
                        sim_steer = float(rp["steering_angle"])
                        print(f"  sim steering: {sim_steer} (diff = {sim_steer - steering_mapped:.4f})")
                    except Exception:
                        pass
                if "speed" in rp:
                    try:
                        sim_speed = float(rp["speed"])
                        print(f"  sim speed:    {sim_speed} (diff = {sim_speed - speed_mapped:.4f})")
                    except Exception:
                        pass
            except Exception as e:
                print("  Could not compute local mapping (agent_params format issue):", e)
        else:
            print("  Skipping local mapping compare (no agent_params.json)")

        # try to recompute reward locally using configs.reward_function.reward_function
        if reward_fn is not None:
            try:
                reward_local = reward_fn(rp)
                print("  local recomputed reward:", reward_local, " (diff env-local = {:.6f})".format(reward_env - reward_local))
            except Exception as e:
                print("  Failed to compute local reward:", e)
        else:
            print("  No local reward function available to compare.")

        # Render a single frame (may return array)
        try:
            frame = env.render()
            if isinstance(frame, np.ndarray):
                print("  render() returned rgb array shape:", frame.shape)
            else:
                print("  render() returned:", type(frame))
        except Exception as e:
            print("  render() error:", e)

        # small pause
        time.sleep(0.5)

        if terminated or truncated:
            print("Episode ended during test; resetting env")
            obs, _ = env.reset()

    env.close()
    print_section("DONE")
    print("If the simulated steering/speed in reward_params matches the locally computed mapping and the local reward is close to the environment reward, the simulator is using your configs as expected.")

if __name__ == "__main__":
    main()

