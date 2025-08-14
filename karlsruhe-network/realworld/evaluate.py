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
EP_LENGTH_S = 2500
EP_SEEDS    = [12345, 67890, 13579, 24680, 11223, 44556, 77889, 99100, 31415, 27182]
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

    #print(f"[DEBUG] Loading VecNormalize from {vecnorm_path}")
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    #print(f"[DEBUG] Loading PPO model from {model_path}")
    model = PPO.load(model_path, env=env, device="cpu")
    return model, env

# ----- Rollout -----
def rollout(model, env):
    #print(f"[DEBUG] Starting rollout...")
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

    #print(f"[DEBUG] Rollout finished after {step_count} steps with reward {ep_reward:.2f}")
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

def shorten_key(orig_key: str) -> str:
    """
    Kürzt SUMO-Systemmetriken für TensorBoard-Logs.
    - 'system_total_waiting_time' → 'mean_waiting_time'
    - 'system_mean_speed' → 'mean_speed'
    - andere Keys bleiben gleich
    """
    if orig_key.startswith("system_"):
        short_key = orig_key[len("system_"):]   # z.B. 'total_waiting_time'
        if short_key.startswith("total_"):
            short_key = short_key[len("total_"):]  # 'waiting_time'
        short_key = "mean_" + short_key            # 'mean_waiting_time'
        return short_key
    else:
        return orig_key

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
    info_acc = []
    ep_reward = 0.0
    step_count = 0

    # gültige Dummy-Aktion aus dem Action Space
    dummy_action = np.array([env.action_space.sample() for _ in range(env.num_envs)])

    while not dones[0]:
        obs, rewards, dones, infos = env.step(dummy_action)
        step_count += 1
        ep_reward += rewards[0] if rewards is not None else 0.0

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
def evaluate():
    results = []
    log_dir_root = os.path.join("evaluation", "logs")
    total_episodes = len(RUNS) * len(SCENARIOS) * N_EPISODES * 3
    ep_counter = 0

    for run_i, run_dir in enumerate(RUNS):
        for sc in SCENARIOS:
            log_dir = os.path.join(
                log_dir_root,
                f"eval_run{run_i}_{os.path.basename(run_dir)}_{sc['name']}"
            )
            os.makedirs(log_dir, exist_ok=True)
            logger = configure(log_dir, ["tensorboard", "stdout"])

            print(f"[INFO] Evaluating run={run_dir}, scenario={sc['name']}")

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
                    "method": "Baseline_FixedTime",
                    "run_dir": os.path.basename(run_dir)
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
                    "method": "Baseline_Actuated",
                    "run_dir": os.path.basename(run_dir)
                })
                results.append(m)

                # --- 3) RL ---
                env_raw = make_env(sc["route_file"], sumo_seed=ep_seed)
                model, env = load_model_and_norm(env_raw, run_dir)
                ep_counter += 1
                print(f"[PROGRESS] RL | {sc['name']} | Ep {ep+1}/{N_EPISODES} "
                      f"({ep_counter}/{total_episodes})")
                m = rollout(model, env)
                m.update({
                    "scenario": sc["name"],
                    "ep_seed": ep_seed,
                    "episode": ep,
                    "method": "RL",
                    "run_dir": os.path.basename(run_dir)
                })
                results.append(m)

                # Logging für alle drei Varianten
                for entry in results[-3:]:
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
