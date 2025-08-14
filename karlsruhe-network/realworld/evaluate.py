import os, json, numpy as np, torch, datetime
import glob
import traci
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
EP_LENGTH_S = 5000
SEEDS       = [546456, 678678, 234256, 678, 10101]
SCENARIOS   = [
    {"name": "morning_peak", "route_file": "flows_morning.rou.xml"},
    {"name": "evening_peak", "route_file": "flows_evening.rou.xml"},
    {"name": "uniform",      "route_file": "flows_uniform.rou.xml"},
    {"name": "random_heavy", "route_file": "flows_random_heavy.rou.xml"},
]

# ====== Gemeinsamer Step-Cache ======
_STEP_CACHE = {"rw_step": None, "queue": 0, "flow": 0}

# ====== Reward-Funktion: RealWorld-Variante ======
def realworld_reward(traffic_signal):
    """
    Belohnt Verkehrsfluss, bestraft Stau und häufige Phasenwechsel.
    Nutzt Exponentielles gleitendes Mittel (EMA) zur Glättung der Messwerte.
    Mit einfachem Cache, um TraCI-Calls pro Step zu reduzieren.
    """
    current_step = int(traci.simulation.getTime())
    if _STEP_CACHE["rw_step"] != current_step:
        _STEP_CACHE["queue"] = int(traffic_signal.get_total_queued())
        _STEP_CACHE["flow"] = int(traci.simulation.getArrivedNumber())
        _STEP_CACHE["rw_step"] = current_step

    q = _STEP_CACHE["queue"]
    f = _STEP_CACHE["flow"]

    # Reset interner Zustände zu Beginn einer Episode
    if hasattr(traffic_signal, "step_count") and traffic_signal.step_count == 0:
        traffic_signal._rw_state = None

    # Initialisierung beim ersten Step
    if not hasattr(traffic_signal, "_rw_state") or traffic_signal._rw_state is None:
        traffic_signal._rw_state = {"prev_q": q, "ema_q": float(q), "ema_f": float(f)}
        return 0.0

    # Parameter für Normalisierung und Gewichtung
    max_storage = 40               # Maximale Speicherkapazität der Kreuzung (Fahrzeuge)
    max_outflow_per_step = 8       # Maximaler Ausfluss pro Step
    w_q, w_build, w_flow, w_switch = 1.0, 0.8, 0.7, 0.1
    ema, clip = 0.3, 5.0           # EMA-Faktor und Reward-Clipping

    # Aktuelle Messwerte
    phase_sw = 1.0 if getattr(traffic_signal, "phase_changed", False) else 0.0
    st = traffic_signal._rw_state

    # EMA-Glättung für Queue und Fluss
    ema_q = (1 - ema) * st["ema_q"] + ema * q
    ema_f = (1 - ema) * st["ema_f"] + ema * f

    # Aufbau neuer Warteschlangen
    delta_q = q - st["prev_q"]
    build = max(0, delta_q)

    # Normalisierung der Werte
    q_norm = np.clip(ema_q / max(1.0, float(max_storage)), 0.0, 1.5)
    b_norm = np.clip(build / max(1.0, float(max_storage) * 0.2), 0.0, 1.5)
    f_norm = np.clip(ema_f / max(1.0, float(max_outflow_per_step)), 0.0, 1.5)

    # Reward-Berechnung
    r = -w_q*q_norm - w_build*b_norm + w_flow*f_norm - w_switch*phase_sw
    r = float(np.clip(r, -clip, clip))

    # Update interner Zustände
    st["prev_q"], st["ema_q"], st["ema_f"] = q, ema_q, ema_f
    traffic_signal._rw_state = st

    return r

# ----- Env Factory -----
def make_env(route_file, sumo_seed):
    print(f"[DEBUG] Creating SUMO env with route={route_file}, seed={sumo_seed}")
    env = parallel_env(
        net_file="map.net.xml",
        route_file=route_file,
        use_gui=False,
        num_seconds=EP_LENGTH_S,
        reward_fn=realworld_reward,
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

    print(f"[DEBUG] Loading VecNormalize from {vecnorm_path}")
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    print(f"[DEBUG] Loading PPO model from {model_path}")
    model = PPO.load(model_path, env=env, device="cpu")
    return model, env

# ----- Rollout -----
# ----- Rollout -----
def rollout(model, env):
    print(f"[DEBUG] Starting rollout...")
    obs = env.reset()
    dones = [False]
    info_acc = []
    step_count = 0
    ep_reward = 0.0

    while not dones[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        step_count += 1
        ep_reward += rewards[0]   # Reward summieren

        if infos:
            info = infos[0] if isinstance(infos, list) else infos
            filtered = {k: float(v) for k, v in info.items() if isinstance(v, (int, float))}
            info_acc.append(filtered)

    print(f"[DEBUG] Rollout finished after {step_count} steps with reward {ep_reward:.2f}")
    mean_metrics = {}
    if info_acc:
        keys = set().union(*[set(d.keys()) for d in info_acc])
        for k in keys:
            vals = [d[k] for d in info_acc if k in d]
            if vals:
                mean_metrics[k] = float(np.mean(vals))

    # Episodenstatistik wie VecMonitor
    mean_metrics["ep_rew"] = ep_reward
    mean_metrics["ep_len"] = step_count

    return mean_metrics

# ----- Evaluation Loop -----
def evaluate():
    results = []
    log_dir = os.path.join("evaluation", "logs",
                           f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["tensorboard", "stdout"])

    for run_dir in RUNS:
        print(f"[INFO] Evaluating run: {run_dir}")
        for sc in SCENARIOS:
            for seed in SEEDS:
                print(f"[INFO] Scenario={sc['name']} | Seed={seed}")
                env_raw = make_env(sc["route_file"], sumo_seed=seed)
                model, env = load_model_and_norm(env_raw, run_dir)

                for ep in range(N_EPISODES):
                    print(f"[DEBUG] Episode {ep+1}/{N_EPISODES} starting...")
                    m = rollout(model, env)
                    m.update({
                        "scenario": sc["name"],
                        "seed": seed,
                        "episode": ep,
                        "method": "RL",
                        "run_dir": os.path.basename(run_dir)
                    })
                    results.append(m)

                    for k, v in m.items():
                        if isinstance(v, (int, float)):
                            logger.record(f"{m['method']}/{sc['name']}/{k}", v)
                    logger.record("rollout/ep_rew_mean", m["ep_rew"])
                    logger.record("rollout/ep_len", m["ep_len"])
                    logger.dump(step=len(results))
                    print(f"[DEBUG] Episode {ep+1} finished. Metrics: {m}")

                env.close()

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
