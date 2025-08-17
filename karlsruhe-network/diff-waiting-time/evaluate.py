import os, json, numpy as np
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.logger import configure
from sumo_rl.environment.env import parallel_env
from supersuit import pad_observations_v0, pad_action_space_v0
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

# ----- Config -----
RUNS = sorted(glob.glob(os.path.join("runs", "ppo_sumo_*")))
MODEL_NAME  = "best_model.zip"
N_EPISODES  = 10
EP_LENGTH_S = 4096
EP_SEEDS    = [12345, 67890, 13579, 24680, 11223, 44556, 77889, 99100, 31415, 27182]
SCENARIOS   = [
    {"name": "morning_peak", "route_file": "flows_morning.rou.xml"},
    {"name": "evening_peak", "route_file": "flows_evening.rou.xml"},
    {"name": "uniform",      "route_file": "flows_uniform.rou.xml"},
    {"name": "random_heavy", "route_file": "flows_random_heavy.rou.xml"},
]

# ----- Env Factory -----
def make_env(route_file, sumo_seed):
    print(f"[DEBUG] Creating SUMO env with route={route_file}, seed={sumo_seed}")
    env = parallel_env(
        net_file="map.net.xml",
        route_file=route_file,
        use_gui=False,
        num_seconds=EP_LENGTH_S,
        reward_fn=dummy_reward,
        min_green=5,
        max_depart_delay=100,
        sumo_seed=sumo_seed,
        add_system_info=True,
        add_per_agent_info=False,
    )
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env

# ----- Model Loader -----
def load_model_and_norm(env, run_dir):
    vecnorm_path = os.path.join(run_dir, "vecnormalize.pkl")
    model_path   = os.path.join(run_dir, MODEL_NAME)

    #print(f"[DEBUG] Loading VecNormalize from {vecnorm_path}")
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    #print(f"[DEBUG] Loading PPO model from {model_path}")
    model = PPO.load(model_path, env=env, device="cpu")
    return model, env

# ----- Rollout -----
def rollout(model, env):
    obs = env.reset()
    dones = [False]
    step_count = 0
    ep_reward = 0.0

    sums = {}
    counts = {}
    last_vals = {}  # nur letzten Wert für system_total_* merken

    while not dones[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        step_count += 1
        ep_reward += rewards[0]

        if infos:
            info = infos[0] if isinstance(infos, list) else infos
            for key, v in info.items():
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    continue
                if key.startswith("system_mean_"):
                    sums[key] = sums.get(key, 0.0) + float(v)
                    counts[key] = counts.get(key, 0) + 1
                elif key.startswith("system_total_"):
                    last_vals[key] = float(v)  # nur letzten Wert behalten

    mean_metrics = {}
    for k, total in sums.items():
        cnt = max(1, counts.get(k, 0))
        mean_metrics[k] = total / cnt

    # Endstände (letzter Wert)
    for k, v in last_vals.items():
        mean_metrics[k] = v

    mean_metrics["ep_rew"] = ep_reward
    mean_metrics["ep_len"] = step_count
    return mean_metrics


def shorten_key(orig_key: str) -> str:
    return orig_key.replace("system_", "")

# ----- Env Factory für Baselines -----
def make_env_baseline(route_file, sumo_seed, fixed_time=True):
    """
    Erstellt eine SUMO-Umgebung, die den internen Controller verwendet.
    fixed_time=True  -> Fester Phasenplan aus net.xml
    fixed_time=False -> SUMO Actuated Control (falls in net.xml konfiguriert)
    """
    env = parallel_env(
        net_file="map.net.xml",
        route_file=route_file,
        use_gui=False,
        num_seconds=EP_LENGTH_S,
        reward_fn=dummy_reward,            # Kein RL-Reward
        fixed_ts=fixed_time,       # True = fixed, False = actuated
        sumo_seed=sumo_seed,
        add_system_info=True,
        add_per_agent_info=False,
    )
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env

def dummy_reward(_ts):
    return 0.0

def rollout_baseline(env):
    obs = env.reset()
    dones = [False]
    ep_reward = 0.0
    step_count = 0

    sums = {}
    counts = {}
    last_vals = {}

    dummy_action = np.array([env.action_space.sample() for _ in range(env.num_envs)])

    while not dones[0]:
        obs, rewards, dones, infos = env.step(dummy_action)
        step_count += 1
        ep_reward += rewards[0] if rewards is not None else 0.0

        if infos:
            info = infos[0] if isinstance(infos, list) else infos
            for key, v in info.items():
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    continue
                if key.startswith("system_mean_"):
                    sums[key] = sums.get(key, 0.0) + float(v)
                    counts[key] = counts.get(key, 0) + 1
                elif key.startswith("system_total_"):
                    last_vals[key] = float(v)

    mean_metrics = {}
    for k, total in sums.items():
        cnt = max(1, counts.get(k, 0))
        mean_metrics[k] = total / cnt

    for k, v in last_vals.items():
        mean_metrics[k] = v

    mean_metrics["ep_rew"] = ep_reward
    mean_metrics["ep_len"] = step_count
    return mean_metrics


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

# ----- Evaluation Loop -----
# ----- Evaluation Loop -----
def evaluate():
    results = []
    log_dir_root = os.path.join("evaluation", "logs")

    # Zählung: 2 Baselines + len(RUNS) RL pro (scenario × episode)
    total_episodes = (2 + len(RUNS)) * len(SCENARIOS) * N_EPISODES
    ep_counter = 0

    for sc in SCENARIOS:
        scen_log_dir = os.path.join(log_dir_root, f"eval_{sc['name']}")
        os.makedirs(scen_log_dir, exist_ok=True)
        logger = configure(scen_log_dir, ["tensorboard", "stdout"])

        print(f"[INFO] Evaluating scenario={sc['name']}")

        for ep in range(N_EPISODES):
            ep_seed = EP_SEEDS[ep]

            # --- 1) Fixed-Time ---
            env = make_env_baseline(sc["route_file"], sumo_seed=ep_seed, fixed_time=True)
            ep_counter += 1
            print(f"[PROGRESS] FixedTime | {sc['name']} | Ep {ep+1}/{N_EPISODES} "
                  f"({ep_counter}/{total_episodes})")
            m = rollout_baseline(env)
            m.update({
                "scenario": sc["name"],
                "ep_seed": ep_seed,
                "episode": ep,
                "method": "Baseline_FixedTime"
            })
            results.append(m)

            # --- 2) Actuated ---
            env = make_env_baseline(sc["route_file"], sumo_seed=ep_seed, fixed_time=False)
            ep_counter += 1
            print(f"[PROGRESS] Actuated | {sc['name']} | Ep {ep+1}/{N_EPISODES} "
                  f"({ep_counter}/{total_episodes})")
            m = rollout_baseline(env)
            m.update({
                "scenario": sc["name"],
                "ep_seed": ep_seed,
                "episode": ep,
                "method": "Baseline_Actuated"
            })
            results.append(m)

            # --- 3) RL-Modelle ---
            for run_dir in RUNS:
                env_raw = make_env(sc["route_file"], sumo_seed=ep_seed)
                model, env = load_model_and_norm(env_raw, run_dir)
                ep_counter += 1
                model_name = os.path.basename(run_dir)
                print(f"[PROGRESS] RL | {sc['name']} | {model_name} "
                      f"| Ep {ep+1}/{N_EPISODES} ({ep_counter}/{total_episodes})")
                m = rollout(model, env)
                
                # Seed extrahieren (3. Teil vom Namen)
                parts = model_name.split("_")
                model_seed = parts[2] if len(parts) > 2 else "unknown"

                m.update({
                    "scenario": sc["name"],
                    "episode": ep,
                    "method": f"{model_name}_{model_seed}"
                })
                results.append(m)

            # --- Logging dieser Episode (Baselines + alle RL) ---
            for entry in results[-(2 + len(RUNS)):]:
                for k, v in entry.items():
                    if isinstance(v, (int, float)) and k not in ["ep_rew", "ep_len", "episode", "ep_seed"]:
                        short_key = shorten_key(k)
                        logger.record(f"{entry['method']}/{short_key}", v)
                logger.record(f"{entry['method']}/ep_rew_mean", entry["ep_rew"])
                logger.record(f"{entry['method']}/ep_len", entry["ep_len"])
            logger.dump(step=ep)

    results_path = os.path.join("evaluation", "eval_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=to_serializable)

    print(f"[INFO] Evaluation abgeschlossen. Ergebnisse: {results_path}")

if __name__ == "__main__":
    evaluate()
