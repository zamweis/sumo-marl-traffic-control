import os
import re
import time
import datetime
import traci
import numpy as np
import torch
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from sumo_rl.environment.env import parallel_env
from supersuit import (
    pad_observations_v0,
    pad_action_space_v0,
    pettingzoo_env_to_vec_env_v1,
    concat_vec_envs_v1
)
import gymnasium as gym
from gymnasium import Wrapper

# ==== Seeds definieren ====
SEEDS = [143534, 456, 635768, 13755]  # beliebig erweiterbar

# ==== RealWorld Reward Function ====
def realworld_reward(traffic_signal):
    # Reset bei Episodenstart
    if hasattr(traffic_signal, "step_count") and traffic_signal.step_count == 0:
        traffic_signal._rw_state = None

    # Lazy init
    if not hasattr(traffic_signal, "_rw_state") or traffic_signal._rw_state is None:
        q = int(traffic_signal.get_total_queued())              # statt get_local_queue()
        f = int(traci.simulation.getArrivedNumber())            # Proxy für Outflow
        traffic_signal._rw_state = {"prev_q": q, "ema_q": float(q), "ema_f": float(f)}
        return 0.0

    # Parameter
    max_storage = 40
    max_outflow_per_step = 8
    w_q, w_build, w_flow, w_switch = 1.0, 0.8, 0.7, 0.1
    ema, clip = 0.3, 5.0

    # Messungen mit vorhandenen APIs
    q = int(traffic_signal.get_total_queued())
    f = int(traci.simulation.getArrivedNumber())                # globaler Durchsatz-Proxy
    phase_sw = 1.0 if getattr(traffic_signal, "phase_changed", False) else 0.0

    st = traffic_signal._rw_state

    # EMA
    ema_q = (1 - ema) * st["ema_q"] + ema * q
    ema_f = (1 - ema) * st["ema_f"] + ema * f

    # Queue-Aufbau
    delta_q = q - st["prev_q"]
    build = max(0, delta_q)

    # Normierung
    q_norm = np.clip(ema_q / max(1.0, float(max_storage)), 0.0, 1.5)
    b_norm = np.clip(build / max(1.0, float(max_storage) * 0.2), 0.0, 1.5)
    f_norm = np.clip(ema_f / max(1.0, float(max_outflow_per_step)), 0.0, 1.5)

    # Reward
    r = -w_q*q_norm - w_build*b_norm + w_flow*f_norm - w_switch*phase_sw
    r = float(np.clip(r, -clip, clip))

    # State-Update
    st["prev_q"], st["ema_q"], st["ema_f"] = q, ema_q, ema_f
    traffic_signal._rw_state = st
    return r

# ==== Custom Reward Function (balanced, bounded to [-1, 1]) ====
def custom_reward(traffic_signal):
    # Lazy init von Zustandsgrößen für Delta-Berechnungen
    if not hasattr(traffic_signal, "_prev_queue"):
        traffic_signal._prev_queue = 0
    if not hasattr(traffic_signal, "_prev_wait_sum"):
        traffic_signal._prev_wait_sum = 0.0

    # --- Metriken aus der SUMO/Env ---
    queue = traffic_signal.get_total_queued()  # #Fahrzeuge im Stau (aktuell)
    wait_sum_total = float(np.sum(traffic_signal.get_accumulated_waiting_time_per_lane()))  # kumuliert seit Sim-Start

    sim = traci.simulation
    arrived = sim.getArrivedNumber()                  # in diesem Step angekommen
    teleports = sim.getStartingTeleportNumber()       # in diesem Step teleportiert
    collisions = sim.getCollidingVehiclesNumber()     # in diesem Step kollidiert

    # --- Deltas (per Step) statt kumulativer Summen ---
    delta_queue = queue - traffic_signal._prev_queue
    # Wartezuwachs seit letztem Step (>= 0 in der Praxis, robust gegen Sprünge)
    delta_wait = max(0.0, wait_sum_total - traffic_signal._prev_wait_sum)

    # Zustände aktualisieren
    traffic_signal._prev_queue = queue
    traffic_signal._prev_wait_sum = wait_sum_total

    # --- Sanfte, robuste Normalisierung via tanh(x / scale) ---
    # Skalen auf praxisnahe Größen einstellen (je nach Netzgröße anpassbar)
    q_term       = -np.tanh(queue      / 30.0)                 # mehr Stau -> stärker negativ
    dq_term      = -np.tanh(max(0,delta_queue) / 10.0)         # nur Stauzunahme bestrafen
    wait_term    = -np.tanh(delta_wait / 60.0)                 # ~60 s zusätzl. Warten → deutliche Strafe
    arrived_term =  np.tanh(arrived    / 5.0)                  # Durchsatz positiv (5 Ankünfte ≈ stark positiv)
    tp_term      = -np.tanh(teleports  / 1.0)                  # schon 1 Teleport → fast maximale Strafe
    col_term     = -np.tanh(collisions / 1.0)                  # schon 1 Kollision → fast maximale Strafe

    # --- Gewichte (balanciert: Stabilität, Fluss, Sicherheit/Fehler) ---
    # Sicherheit/Fehler am stärksten, dann Wartezeiten/Stau, dann Durchsatz
    w_q, w_dq, w_wait = 0.35, 0.20, 0.45
    w_arr, w_tp, w_col = 0.40, 0.90, 1.20

    raw = (
        w_q   * q_term +
        w_dq  * dq_term +
        w_wait* wait_term +
        w_arr * arrived_term +
        w_tp  * tp_term +
        w_col * col_term
    )

    # Optionale harte Zusatzstrafen bei grobem Fehlverhalten
    if teleports > 0:
        raw -= 0.5
    if collisions > 0:
        raw -= 0.8

    # Gesamtreward stabil in (-1, 1) abbilden
    reward = float(np.tanh(raw))

    return reward

# ==== Adaptive Parameter-Schedules ====
def adaptive_entropy_schedule(start=0.01):
    return lambda progress: max(0.001, start * (1 - progress))

def dynamic_clip_range(start=0.2):
    return lambda progress: max(0.1, start * (1 - 0.5 * progress))

def linear_schedule(start):
    return lambda progress: start * (1 - progress)

def cosine_warmup_floor(start=3e-4, warmup_frac=0.05, min_lr_frac=0.1):
    """
    progress_remaining: 1 -> 0  (SB3-Konvention)
    """
    min_lr = start * min_lr_frac
    warmup_frac = max(0.0, min(0.5, warmup_frac))  # begrenzen

    def schedule(progress_remaining: float) -> float:
        t = 1.0 - progress_remaining  # 0..1 (vergangener Anteil)
        if t < warmup_frac:
            # linearer Warmup von 0.1*start auf start
            base = start * 0.1 + (start - start * 0.1) * (t / warmup_frac)
        else:
            # Cosine von start -> min_lr
            tt = (t - warmup_frac) / max(1e-8, (1.0 - warmup_frac))
            cos_term = 0.5 * (1 + np.cos(np.pi * tt))
            base = min_lr + (start - min_lr) * cos_term
        return float(base)
    return schedule

def two_phase_linear(start=3e-4, plateau_frac=0.4, min_lr_frac=0.2):
    min_lr = start * min_lr_frac
    plateau_frac = max(0.0, min(0.9, plateau_frac))
    def schedule(progress_remaining: float) -> float:
        t = 1.0 - progress_remaining
        if t < plateau_frac:
            return float(start)
        # linear von start -> min_lr
        tt = (t - plateau_frac) / max(1e-8, (1.0 - plateau_frac))
        return float(start + (min_lr - start) * tt)
    return schedule

def inv_sqrt_schedule(start=3e-4, min_lr_frac=0.15, k=3.0):
    """
    lr = min_lr + (start - min_lr) / sqrt(1 + k * t)
    """
    min_lr = start * min_lr_frac
    def schedule(progress_remaining: float) -> float:
        t = 1.0 - progress_remaining
        return float(min_lr + (start - min_lr) / np.sqrt(1.0 + k * t))
    return schedule

# ==== Finde letzten vollständigen Run ====
def find_latest_complete_run(base_dir="runs", prefix="ppo_sumo_"):
    subdirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith(prefix)],
        reverse=True
    )
    for d in subdirs:
        dir_path = os.path.join(base_dir, d)
        norm_path = os.path.join(dir_path, "vecnormalize.pkl")
        if not os.path.exists(norm_path):
            continue

        final_model = os.path.join(dir_path, "model.zip")
        if os.path.exists(final_model):
            return dir_path, final_model, norm_path

        checkpoint_models = [
            f for f in os.listdir(dir_path)
            if re.match(r"ppo_sumo_model_(\d+)_steps\.zip", f)
        ]
        if checkpoint_models:
            checkpoint_models.sort(key=lambda x: int(re.findall(r"\d+", x)[0]), reverse=True)
            best_checkpoint = checkpoint_models[0]
            return dir_path, os.path.join(dir_path, best_checkpoint), norm_path

    return None

# ==== Checkpoint Callback ====
class TimeBasedCheckpointCallback(BaseCallback):
    def __init__(self, save_interval_sec, save_path, name_prefix="ppo_sumo_model", verbose=0):
        super().__init__(verbose)
        self.save_interval_sec = save_interval_sec
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_save_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval_sec:
            timestep = self.num_timesteps
            filename = f"{self.name_prefix}_{timestep}_steps"
            self.model.save(os.path.join(self.save_path, filename + ".zip"))
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.save_path, f"{filename}_vecnormalize.pkl"))
            print(f"[Checkpoint] Modell gespeichert bei {timestep} Schritten ({filename})")
            self.last_save_time = current_time
        return True

# ==== Env Metric Logger Callback ====
SYSTEM_KEY_MAP = {
    "system_total_running": "mean_running",
    "system_total_backlogged": "mean_backlogged",
    "system_total_stopped": "mean_stopped",
    "system_total_arrived": "mean_arrived",
    "system_total_departed": "mean_departed",
    "system_total_teleported": "mean_teleported",
    "system_total_waiting_time": "mean_waiting_time",
    "system_mean_waiting_time": "mean_waiting_time",
    "system_mean_speed": "mean_speed"
}

class EnvMetricsLoggerCallback(BaseCallback):
    def __init__(self, prefix="env", verbose=0):
        super().__init__(verbose)
        self.prefix = prefix
        self.sums = {}
        self.counts = {}

    def _on_rollout_start(self) -> None:
        # Reset Akkumulatoren am Start des Rollouts
        self.sums.clear()
        self.counts.clear()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue
            for orig_key, short_key in SYSTEM_KEY_MAP.items():
                if orig_key not in info:
                    continue
                v = info[orig_key]
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    continue
                tag = f"{self.prefix}/{short_key}"
                self.sums[tag] = self.sums.get(tag, 0.0) + float(v)
                self.counts[tag] = self.counts.get(tag, 0) + 1
        return True

    def _on_rollout_end(self) -> None:
        # Am Ende des Rollouts Mittelwert loggen
        for tag, total in self.sums.items():
            mean_val = total / max(1, self.counts.get(tag, 1))
            self.logger.record(tag, mean_val)

# ==== Best Model Saver Callback ====
class BestModelSaverCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.best_mean_reward = -float('inf')
        self.save_path = save_path

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        ep_info_buffer = self.model.ep_info_buffer
        if len(ep_info_buffer) > 0:
            mean_rew = np.mean([ep_info['r'] for ep_info in ep_info_buffer])
            if mean_rew > self.best_mean_reward:
                self.best_mean_reward = mean_rew
                model_path = os.path.join(self.save_path, "best_model.zip")
                self.model.save(model_path)
                if hasattr(self.model.env, "save"):
                    norm_path = os.path.join(self.save_path, "best_model_vecnormalize.pkl")
                    self.model.env.save(norm_path)
                print(f"[AUTOLOG] Neuer Bestwert {mean_rew:.2f} → best_model gespeichert!", flush=True)

# ==== Hauptschleife über Seeds ====
for SEED in SEEDS:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", f"ppo_sumo_{SEED}_{now}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n[INFO] Starte Training mit Seed: {SEED}")

    env = parallel_env(
        net_file="map.net.xml",
        route_file="map.rou.xml",
        use_gui=False,
        num_seconds=5000,
        reward_fn=realworld_reward,
        min_green=5,
        max_depart_delay=100,
        sumo_seed=SEED,
        add_system_info=True,
        add_per_agent_info=False,
    )

    if hasattr(env, "seed"):
        env.seed(SEED)

    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=8, base_class="stable_baselines3")
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        batch_size=2048,
        n_steps=2048,
        learning_rate=cosine_warmup_floor(start=3e-4, warmup_frac=0.05, min_lr_frac=0.1),
        clip_range=dynamic_clip_range(0.2),
        ent_coef=0.005,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )

    callbacks = CallbackList([
        TimeBasedCheckpointCallback(
            save_interval_sec=3600,
            save_path=log_dir,
            name_prefix="ppo_sumo_model",
            verbose=1,
        ),
        EnvMetricsLoggerCallback(),
        BestModelSaverCallback(save_path=log_dir),
    ])

    try:
        time.sleep(3) # Für sauber getrennte setuplogs in Console
        model.learn(
            total_timesteps=5_000_000,
            callback=callbacks,
        )
        model.save(os.path.join(log_dir, "model.zip"))
        env.save(os.path.join(log_dir, "vecnormalize.pkl"))
        print(f"\n[INFO] Training abgeschlossen für Seed {SEED}. Modell gespeichert unter: {log_dir}")

    except KeyboardInterrupt:
        print("[ABBRUCH] Manuelles Beenden erkannt. Speichere aktuellen Stand...")
        model.save(os.path.join(log_dir, "model_interrupt.zip"))
        env.save(os.path.join(log_dir, "vecnormalize_interrupt.pkl"))

    except Exception as e:
        print(f"\n[FEHLER] Während des Trainings bei Seed {SEED} aufgetreten: {e}")

    finally:
        try:
            env.save(os.path.join(log_dir, "vecnormalize.pkl"))
        except Exception as e:
            print(f"[WARNUNG] VecNormalize konnte nicht gespeichert werden: {e}")
        env.close()
