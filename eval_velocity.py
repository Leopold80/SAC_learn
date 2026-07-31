import json, os, sys
import numpy as np
from stable_baselines3 import PPO
import sac_experiments.go2_env
import gymnasium as gym

def analyze_model(model_path, label, env):
    model = PPO.load(model_path, env=env, device='cpu')
    N_EPISODES = 10
    MAX_STEPS = 1000

    all_cmd_x, all_cmd_y, all_cmd_yaw = [], [], []
    all_vel_x, all_vel_y, all_vel_yaw = [], [], []
    all_rewards = []

    for ep in range(N_EPISODES):
        obs, info = env.reset()
        ep_rew = 0
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_rew += reward
            all_cmd_x.append(info['commands'][0])
            all_cmd_y.append(info['commands'][1])
            all_cmd_yaw.append(info['commands'][2])
            all_vel_x.append(info['local_linear_velocity'][0])
            all_vel_y.append(info['local_linear_velocity'][1])
            all_vel_yaw.append(info['local_angular_velocity'][2])
            if term or trunc:
                break
        all_rewards.append(ep_rew)

    cmd_x = np.array(all_cmd_x)
    cmd_y = np.array(all_cmd_y)
    cmd_yaw = np.array(all_cmd_yaw)
    vel_x = np.array(all_vel_x)
    vel_y = np.array(all_vel_y)
    vel_yaw = np.array(all_vel_yaw)

    x_error = cmd_x - vel_x
    y_error = cmd_y - vel_y
    yaw_error = cmd_yaw - vel_yaw

    results = {
        "label": label,
        "model": str(model_path),
        "n_episodes": N_EPISODES,
        "forward": {
            "cmd_mean": float(cmd_x.mean()),
            "cmd_std": float(cmd_x.std()),
            "vel_mean": float(vel_x.mean()),
            "vel_std": float(vel_x.std()),
            "bias": float(x_error.mean()),
            "abs_error_mean": float(np.abs(x_error).mean()),
            "rmse": float(np.sqrt(np.mean(x_error**2))),
            "track_ratio": float(vel_x.mean() / max(cmd_x.mean(), 0.001)),
        },
        "lateral": {
            "cmd_mean": float(cmd_y.mean()),
            "cmd_std": float(cmd_y.std()),
            "vel_mean": float(vel_y.mean()),
            "vel_std": float(vel_y.std()),
            "abs_error_mean": float(np.abs(y_error).mean()),
            "rmse": float(np.sqrt(np.mean(y_error**2))),
        },
        "yaw": {
            "cmd_mean": float(cmd_yaw.mean()),
            "cmd_std": float(cmd_yaw.std()),
            "vel_mean": float(vel_yaw.mean()),
            "vel_std": float(vel_yaw.std()),
            "abs_error_mean": float(np.abs(yaw_error).mean()),
            "rmse": float(np.sqrt(np.mean(yaw_error**2))),
        },
        "overall_rmse": float(np.sqrt(np.mean(x_error**2 + y_error**2 + yaw_error**2))),
        "avg_episode_reward": float(np.mean(all_rewards)),
        "std_episode_reward": float(np.std(all_rewards)),
        "speed_bins": [],
    }

    for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (cmd_x >= lo) & (cmd_x < hi)
        if mask.sum() > 0:
            r = (vel_x[mask].mean() / cmd_x[mask].mean()) * 100 if cmd_x[mask].mean() > 0 else 0
            results["speed_bins"].append({
                "range": [lo, hi],
                "n": int(mask.sum()),
                "cmd_mean": float(cmd_x[mask].mean()),
                "vel_mean": float(vel_x[mask].mean()),
                "ratio_pct": round(r, 1),
            })

    return results

# --- Main ---
env = gym.make('Go2Locomotion-v0')

models = [
    ("outputs/go2_ppo_32env_v2/best_model/best_model.zip", "v2 (new reward)"),
    ("outputs/go2_ppo_32env_fixed/best_model/best_model.zip", "v1 (old reward)"),
]

all_results = {}
for path, label in models:
    print(f"\nAnalyzing {label}...")
    r = analyze_model(path, label, env)
    all_results[label] = r

    # Print summary
    fwd = r["forward"]
    print(f"  Forward: cmd={fwd['cmd_mean']:.3f} vel={fwd['vel_mean']:.3f} "
          f"bias={fwd['bias']:.3f} rmse={fwd['rmse']:.3f} ratio={fwd['track_ratio']:.2f}")
    print(f"  Episode reward: {r['avg_episode_reward']:.1f} +/- {r['std_episode_reward']:.1f}")
    for b in r["speed_bins"]:
        print(f"    [{b['range'][0]:.1f},{b['range'][1]:.1f}): cmd={b['cmd_mean']:.3f} vel={b['vel_mean']:.3f} ratio={b['ratio_pct']:.1f}%")

env.close()

# Save
out_path = "outputs/velocity_analysis.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
