import os, json, numpy as np, torch, datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.logger import configure
from sumo_rl.environment.env import parallel_env
from supersuit import pad_observations_v0, pad_action_space_v0
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

# ----- Config -----
RUN_DIR     = os.path.join("runs", "ppo_sumo_..._YYYY-mm-dd_HH-MM-SS")  # Pfad relativ zum training-Ordner
MODEL_NAME  = "best_model.zip"
N_EPISODES  = 10
EP_LENGTH_S = 5000
SEEDS       = [546456, 678678, 234256, 678, 10101]
SCENARIOS   = [
    {"name": "morning_peak", "route_file": "flows_morning.rou.xml"},
    {"name": "evening_peak", "route_file": "flows_evening.rou.xml"},
    {"name": "uniform",      "route_file": "flows_uniform.rou.xml"},
    {"name": "random_heavy", "route_file": "flows_random_heavy.rou.xml"},
]

# ----- Env Factory -----
def make_env(route_file, sumo_seed):
    env = parallel_env(
        net_file="network.net.xml",
        route_file=route_file,
        use_gui=False,
        num_seconds=EP_LENGTH_S,
        reward_fn=None,
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
def load_model_and_norm(env):
    vecnorm_path = os.path.join(RUN_DIR, "vecnormalize.pkl")
    model_path   = os.path.join(RUN_DIR, MODEL_NAME)
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(model_path, env=env, device="cpu")
    return model, env

# ----- Rollout -----
def rollout(model, env):
    obs = env.reset()
    dones = [False]
    info_acc = []
    while not dones[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if infos:
            info = infos[0] if isinstance(infos, list) else infos
            filtered = {k: float(v) for k, v in info.items() if isinstance(v, (int, float))}
            info_acc.append(filtered)
    mean_metrics = {}
    if info_acc:
        keys = set().union(*[set(d.keys()) for d in info_acc])
        for k in keys:
            vals = [d[k] for d in info_acc if k in d]
            if vals:
                mean_metrics[k] = float(np.mean(vals))
    return mean_metrics

# ----- Evaluation Loop -----
def evaluate():
    results = []
    # TensorBoard-Logger im evaluation/logs/ Ordner
    log_dir = os.path.join("evaluation", "logs", f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["tensorboard", "stdout"])

    for sc in SCENARIOS:
        for seed in SEEDS:
            env_raw = make_env(sc["route_file"], sumo_seed=seed)
            model, env = load_model_and_norm(env_raw)

            for ep in range(N_EPISODES):
                m = rollout(model, env)
                m.update({
                    "scenario": sc["name"],
                    "seed": seed,
                    "episode": ep,
                    "method": "RL"   # später für Vergleich mit Baseline nützlich
                })
                results.append(m)

                # Logging in TensorBoard
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        logger.record(f"{m['method']}/{sc['name']}/{k}", v)
                logger.dump(step=len(results))

            env.close()

    # Ergebnisse als JSON in evaluation/
    results_path = os.path.join("evaluation", "eval_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Evaluation abgeschlossen.")
    print(f"      Ergebnisse gespeichert unter: {results_path}")
    print(f"      TensorBoard-Logs gespeichert unter: {log_dir}")


if __name__ == "__main__":
    np.random.seed(0); torch.manual_seed(0)
    evaluate()
