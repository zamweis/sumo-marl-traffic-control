# ====== Bibliotheken und Module ======
# Standard-Module für Dateiverwaltung, Zeit und Regex
import os
import re
import time
import datetime
import random

# SUMO-Interface (TraCI) für Simulation
import traci

# Mathematische und numerische Berechnungen
import numpy as np

# PyTorch für neuronale Netze und Reproduzierbarkeit
import torch

# Stable-Baselines3 (RL-Algorithmen, hier PPO)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# SUMO-RL-Umgebung (PettingZoo-kompatibel)
from sumo_rl.environment.env import parallel_env

# SuperSuit – Hilfsfunktionen, um PettingZoo-Umgebungen mit SB3 zu verwenden
from supersuit import (
    pad_observations_v0,          # Padding für Beobachtungen, um feste Größe zu garantieren
    pad_action_space_v0,          # Padding für Aktionsraum
    pettingzoo_env_to_vec_env_v1, # Konvertierung zu SB3-kompatiblem Vektor-Env
    concat_vec_envs_v1            # Mehrere Envs parallel laufen lassen
)

# Gymnasium für RL-Umgebungs-Schnittstellen
import gymnasium as gym
from gymnasium import Wrapper


# ====== Trainings-Setup ======
SEEDS = [143534, 456, 635768, 13755]  # Verschiedene Zufalls-Seed-Werte für reproduzierbare Runs
ROUTE_FILES = [
    "flows_evening.rou.xml",
    "flows_morning.rou.xml",
    "flows_uniform.rou.xml",
]

# ====== Schedules für Hyperparameter-Anpassung ======
# (Funktionen, die während des Trainings den Wert z. B. von Lernrate oder Clip-Bereich dynamisch anpassen)
def adaptive_entropy_schedule(start=0.01):
    return lambda progress: max(0.001, start * (1 - progress))

def dynamic_clip_range(start=0.2, end=0.1):
    return lambda pr: end + (start - end) * pr

def cosine_clip(start=0.2, end=0.1):
    return lambda pr: end + (start - end) * 0.5 * (1 + np.cos(np.pi * (1 - pr)))

def linear_schedule(start):
    return lambda progress: start * (1 - progress)

def cosine_warmup_floor(start=3e-4, warmup_frac=0.05, min_lr_frac=0.1):
    """
    Lernrate: Erst linear hochfahren (Warmup), dann mit Cosinus auf Minimalwert absenken.
    """
    min_lr = start * min_lr_frac
    warmup_frac = max(0.0, min(0.5, warmup_frac))
    def schedule(progress_remaining: float) -> float:
        t = 1.0 - progress_remaining
        if t < warmup_frac:
            base = start * 0.1 + (start - start * 0.1) * (t / warmup_frac)
        else:
            tt = (t - warmup_frac) / max(1e-8, (1.0 - warmup_frac))
            cos_term = 0.5 * (1 + np.cos(np.pi * tt))
            base = min_lr + (start - min_lr) * cos_term
        return float(base)
    return schedule

# ====== Hilfsfunktionen und Callbacks ======
# (Modelle finden, Checkpoints speichern, Metriken loggen, bestes Modell sichern)
# ====== Letzten vollständigen Run finden ======
def find_latest_complete_run(base_dir="runs", prefix="ppo_sumo_"):
    """
    Sucht im 'runs'-Ordner nach dem neuesten Trainingslauf, der
    - eine gespeicherte VecNormalize-Instanz hat
    - und entweder ein finales Modell oder mindestens einen Checkpoint.
    Gibt die Pfade zu Run-Ordner, Modell und Normalisierungsdatei zurück.
    """
    subdirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith(prefix)],
        reverse=True
    )
    for d in subdirs:
        dir_path = os.path.join(base_dir, d)
        norm_path = os.path.join(dir_path, "vecnormalize.pkl")
        if not os.path.exists(norm_path):
            continue

        # Prüfe auf finales Modell
        final_model = os.path.join(dir_path, "model.zip")
        if os.path.exists(final_model):
            return dir_path, final_model, norm_path

        # Falls kein finales Modell: Prüfe auf Checkpoints
        checkpoint_models = [
            f for f in os.listdir(dir_path)
            if re.match(r"ppo_sumo_model_(\d+)_steps\.zip", f)
        ]
        if checkpoint_models:
            checkpoint_models.sort(key=lambda x: int(re.findall(r"\d+", x)[0]), reverse=True)
            best_checkpoint = checkpoint_models[0]
            return dir_path, os.path.join(dir_path, best_checkpoint), norm_path

    return None

def make_env(seed, route_files):
    def _init():
        env = parallel_env(
            net_file="map.net.xml",
            route_file=route_files[0],  # Platzhalter
            use_gui=False,
            num_seconds=4096,
            reward_fn="queue",
            min_green=5,
            max_depart_delay=100,
            sumo_seed=seed,
            add_system_info=True,
            add_per_agent_info=False,
        )
        if hasattr(env, "seed"):
            env.seed(seed)

        orig_reset = env.reset
        idx = {"i": -1}  # mutierbares Zähl-Objekt im Closure

        def reset_with_round_robin(**kwargs):
            idx["i"] = (idx["i"] + 1) % len(route_files)
            new_route = route_files[idx["i"]]
            env.route_file = new_route
            if hasattr(env, "sumo_seed"):
                env.sumo_seed = seed
            print(f"\n[DEBUG] Reset → Route: {new_route} | Seed: {seed}\n", flush=True)
            return orig_reset(**kwargs)

        env.reset = reset_with_round_robin
        return env
    return _init


def shorten_key(orig_key: str) -> str:
    return orig_key.replace("system_", "")

# ====== Callback: Zeitbasiertes Speichern ======
class TimeBasedCheckpointCallback(BaseCallback):
    """
    Speichert Modell und Normalisierungsdaten in festen Zeitintervallen (Sekunden).
    """
    def __init__(self, save_interval_sec, save_path, name_prefix="ppo_sumo_model", verbose=0):
        super().__init__(verbose)
        self.save_interval_sec = save_interval_sec
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_save_time = time.time()

    def _on_step(self) -> bool:
        return True  # Keine Aktion bei jedem einzelnen Step

    def _on_rollout_end(self) -> bool:
        # Am Ende eines Rollouts prüfen, ob das Zeitintervall abgelaufen ist
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


# ====== Callback: Metriken aus der Env loggen ======
class EpisodeMetricsLoggerCallback(BaseCallback):
    def __init__(self, prefix="episode", verbose=0):
        super().__init__(verbose)
        self.prefix = prefix
        self.verbose = verbose
        self.sums = {}
        self.counts = {}
        self.last_totals = {}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if infos is None:
            return True

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            if dones is not None and dones[i]:
                # --- Episode zu Ende ---
                fin = info.get("final_info") or info.get("terminal_info")
                if isinstance(fin, dict):
                    for k, v in fin.items():
                        if not isinstance(v, (int, float)) or not np.isfinite(v):
                            continue
                        if k.startswith("system_mean_"):
                            self.sums[k] = self.sums.get(k, 0.0) + float(v)
                            self.counts[k] = self.counts.get(k, 0) + 1
                        elif k.startswith("system_total_"):
                            self.last_totals[k] = float(v)
            else:
                # --- Nur Zwischenschritt, solange Episode noch läuft ---
                for k, v in info.items():
                    if not isinstance(v, (int, float)) or not np.isfinite(v):
                        continue
                    if k.startswith("system_mean_") or k in [
                        "system_total_waiting_time",
                        "system_total_stopped",
                        "system_total_running",
                    ]:
                        self.sums[k] = self.sums.get(k, 0.0) + float(v)
                        self.counts[k] = self.counts.get(k, 0) + 1
                    elif k.startswith("system_total_"):
                        self.last_totals[k] = float(v)

        # Episode fertig → loggen
        if dones is not None and any(dones):
            for k, total in self.sums.items():
                mean_val = total / max(1, self.counts.get(k, 1))
                short_key = shorten_key(k)
                self.logger.record(f"{self.prefix}/{short_key}", mean_val)
                if self.verbose:
                    print(f"[EpisodeMetrics] {short_key} (mean) = {mean_val:.3f}")

            for k, v in self.last_totals.items():
                short_key = shorten_key(k)
                self.logger.record(f"{self.prefix}/{short_key}", v)
                if self.verbose:
                    print(f"[EpisodeMetrics] {short_key} (total) = {v:.0f}")

            # Reset für nächste Episode
            self.sums.clear()
            self.counts.clear()
            self.last_totals.clear()

        return True

# ====== Callback: Bestes Modell speichern ======
class BestModelSaverCallback(BaseCallback):
    """
    Speichert das Modell mit dem bisher höchsten mittleren Episodenreward.
    """
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


# ====== Haupt-Trainingsschleife ======
for SEED in SEEDS:
    # Reproduzierbarkeit sicherstellen
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Log-Verzeichnis erstellen
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", f"ppo_sumo_{SEED}_{now}")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n[INFO] Starte Training mit Seed: {SEED}")

    # SUMO-Umgebung initialisieren
    env = make_env(SEED, ROUTE_FILES)()

    # Falls die Env einen seed()-Aufruf unterstützt
    if hasattr(env, "seed"):
        env.seed(SEED)

    # Anpassung der Beobachtungen und Aktionen an SB3
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = pettingzoo_env_to_vec_env_v1(env)

    # WICHTIG: trotzdem concat_vec_envs_v1 mit num_vec_envs=1
    env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")


    # Logging und Normalisierung
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # PPO-Agent erstellen
    model = PPO(
        policy="MlpPolicy",      # Mehrschicht-Perzeptron-Policy
        env=env,
        verbose=1,               # Ausführliches Logging
        tensorboard_log=log_dir, # TensorBoard-Pfad
        batch_size=256,          # Minibatch-Größe für PPO
        n_steps=2048,            # Rollout-Länge
        learning_rate=cosine_warmup_floor(start=3e-4, warmup_frac=0.05, min_lr_frac=0.1),
        clip_range=cosine_clip(), # Clipping-Range dynamisch
        ent_coef=0.01,            # Entropie-Koeffizient (Exploration)
        gamma=0.99,               # Diskontfaktor
        gae_lambda=0.95,          # Lambda für GAE
        device="cpu",             # Training auf CPU
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])), # Netzarchitekturgit
    )

    # Callback-Liste: Checkpoints, Logging, Best-Model-Speicherung
    callbacks = CallbackList([
        TimeBasedCheckpointCallback(
            save_interval_sec=3600, # Jede Stunde speichern
            save_path=log_dir,
            name_prefix="ppo_sumo_model",
            verbose=1,
        ),
        EpisodeMetricsLoggerCallback(),
        BestModelSaverCallback(save_path=log_dir),
    ])

    # Training starten
    try:
        time.sleep(3) # Kurze Pause für saubere Konsolenlogs
        model.learn(
            total_timesteps=2_000_000,
            callback=callbacks,
        )
        # Nach Abschluss final speichern
        model.save(os.path.join(log_dir, "model.zip"))
        env.save(os.path.join(log_dir, "vecnormalize.pkl"))
        print(f"\n[INFO] Training abgeschlossen für Seed {SEED}. Modell gespeichert unter: {log_dir}")

    # Falls Training manuell abgebrochen wird (Strg+C)
    except KeyboardInterrupt:
        print("[ABBRUCH] Manuelles Beenden erkannt. Speichere aktuellen Stand...")
        model.save(os.path.join(log_dir, "model_interrupt.zip"))
        env.save(os.path.join(log_dir, "vecnormalize_interrupt.pkl"))

    # Generelle Fehlerbehandlung
    except Exception as e:
        print(f"\n[FEHLER] Während des Trainings bei Seed {SEED} aufgetreten: {e}")

    # Cleanup: Env schließen und Normalisierungsdaten sichern
    finally:
        try:
            env.save(os.path.join(log_dir, "vecnormalize.pkl"))
        except Exception as e:
            print(f"[WARNUNG] VecNormalize konnte nicht gespeichert werden: {e}")
        env.close()
