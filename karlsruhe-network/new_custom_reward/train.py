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
from gym import Wrapper

# ==== Seeds definieren ====
SEEDS = [143534, 456, 635768, 13755]  # beliebig erweiterbar

# ==== RealWorld Reward Function ====

# ==== Custom Reward Function (real-world measurable only) ====
def realworld_reward(traffic_signal):
    # Reset-Check: Wenn neue Episode beginnt -> State löschen
    if hasattr(traffic_signal, "step_count") and traffic_signal.step_count == 0:
        traffic_signal._rw_state = None

    """
    Reward-Funktion, die nur real messbare Metriken nutzt:
    - Queue (Anzahl Fahrzeuge in Anfahrtsbereichen)
    - Outflow (Fahrzeuge, die in diesem Step die Kreuzung verlassen)
    - Optional: Phasenwechsel (phase_changed-Flag)
    """
    # --- Lazy init von Zustandsgrößen für Deltas & EMA ---
    if not hasattr(traffic_signal, "_rw_state") or traffic_signal._rw_state is None:
        q = int(traffic_signal.get_local_queue())
        f = int(traffic_signal.get_local_outflow())
        traffic_signal._rw_state = {"prev_q": q, "ema_q": float(q), "ema_f": float(f)}
        return 0.0

    # --- Parameter (an Kreuzung anpassen) ---
    max_storage = 40            # max. Fahrzeuge, die sich in allen Zufahrten stauen können
    max_outflow_per_step = 8    # physikalisch möglicher Abfluss pro Step (alle Ausfahrten)
    w_q = 1.0                   # Gewicht: Queue-Niveau strafen
    w_build = 0.8                # Gewicht: Queue-Aufbau strafen
    w_flow = 0.7                 # Gewicht: Durchsatz belohnen
    w_switch = 0.1               # Gewicht: Phasenwechsel leicht strafen
    ema = 0.3                    # EMA-Faktor (0..1); höher = weniger Glättung
    clip = 5.0                   # Clipping für stabilen Wertebereich

    # --- Messungen ---
    q = int(traffic_signal.get_local_queue())
    f = int(traffic_signal.get_local_outflow())
    phase_sw = 1.0 if getattr(traffic_signal, "phase_changed", False) else 0.0

    # --- Vorherige Werte laden ---
    st = traffic_signal._rw_state

    # --- Glättung (EMA) ---
    ema_q = (1 - ema) * st["ema_q"] + ema * q
    ema_f = (1 - ema) * st["ema_f"] + ema * f

    # --- Queue-Aufbau (nur positives Wachstum strafen) ---
    delta_q = q - st["prev_q"]
    build = max(0, delta_q)

    # --- Normierung ---
    q_norm = np.clip(ema_q / max(1.0, float(max_storage)), 0.0, 1.5)
    b_norm = np.clip(build / max(1.0, float(max_storage) * 0.2), 0.0, 1.5)
    f_norm = np.clip(ema_f / max(1.0, float(max_outflow_per_step)), 0.0, 1.5)

    # --- Reward ---
    r = (
        -w_q * q_norm
        -w_build * b_norm
        +w_flow * f_norm
        -w_switch * phase_sw
    )

    # --- Clipping & State Update ---
    r = float(np.clip(r, -clip, clip))
    st["prev_q"] = q
    st["ema_q"] = ema_q
    st["ema_f"] = ema_f
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

# ==== Env Metric Logger ====
class EnvMetricsLoggerCallback(BaseCallback):
    def __init__(self, prefix="env", verbose=0):
        super().__init__(verbose); self.prefix = prefix

    def _flatten_items(self, d, parent=""):
        for k, v in d.items():
            key = f"{parent}.{k}" if parent else str(k)
            if isinstance(v, dict):
                yield from self._flatten_items(v, key)
            else:
                yield key, v

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos: return True
        sums, counts = {}, {}
        for info in infos:
            if not isinstance(info, dict): continue
            for k, v in self._flatten_items(info):
                if isinstance(v, (int, float)) and np.isfinite(v):
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        for k in sums:
            self.logger.record(f"{self.prefix}/{k}", sums[k] / max(1, counts[k]))
        return True

# ==== Learning Rate Logger ====
class LearningRateLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.model._total_timesteps
        lr = self.model.lr_schedule(progress)
        self.logger.record("train/learning_rate", lr)

        if hasattr(self.model, 'clip_range'):
            clip = self.model.clip_range(progress)
            self.logger.record("train/clip_range", clip)

        return True

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
        reward_fn=custom_reward,
        min_green=5,
        max_depart_delay=100,
        sumo_seed=SEED,
        add_system_info=True,
        add_per_agent_info=True, 
    )

    if hasattr(env, "seed"):
        env.seed(SEED)

    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=8, base_class="stable_baselines3")
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        batch_size=2048,
        n_steps=2048,
        learning_rate=linear_schedule(3e-4),
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
        LearningRateLoggerCallback(),
        BestModelSaverCallback(save_path=log_dir),
    ])

    try:
        model.learn(
            total_timesteps=10_000_000,
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
