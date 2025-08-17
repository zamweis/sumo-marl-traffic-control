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


# ====== Reward-Funktion: RealWorld-Variante ======
def realworld_reward(traffic_signal):
    """
    Verbesserte Reward-Funktion:
    - belohnt Outflow stärker
    - bestraft Queue-Aufbau und häufige Phasenwechsel
    - EMA-Glättung für stabilere Werte
    - kleine positive Basis, damit PPO leichter lernt
    """

    # aktuelle Messwerte lokal zur Kreuzung
    q = traffic_signal.get_total_queued()
    f = sum(
        traffic_signal.sumo.lane.getLastStepVehicleNumber(l)
        for l in traffic_signal.out_lanes
    )

    # interne Zustände initialisieren
    if not hasattr(traffic_signal, "_rw_state") or traffic_signal.env.sim_step == 0:
        traffic_signal._rw_state = {"prev_q": q, "ema_q": float(q), "ema_f": float(f)}
        return 0.0

    st = traffic_signal._rw_state

    # Parameter für Normalisierung / Gewichtung
    max_storage = max(5.0, len(traffic_signal.lanes) * 12)   # etwas großzügiger als *10
    max_outflow_per_step = max(2.0, len(traffic_signal.out_lanes) * 3)  # großzügiger als *2
    w_q, w_build, w_flow, w_switch = 1.0, 0.8, 1.0, 0.1      # Flow stärker gewichtet
    ema, clip = 0.3, 2.0                                     # glatter, engerer Clip

    # Exponentielles Mittel
    ema_q = (1 - ema) * st["ema_q"] + ema * q
    ema_f = (1 - ema) * st["ema_f"] + ema * f

    # Aufbau neuer Warteschlangen
    delta_q = q - st["prev_q"]
    build = max(0, delta_q)

    # Normierungen
    q_norm = np.clip(ema_q / max_storage, 0.0, 1.5)
    b_norm = np.clip(build / (0.2 * max_storage), 0.0, 1.5)
    f_norm = np.clip(ema_f / max_outflow_per_step, 0.0, 1.5)

    # Phasenwechsel-Bestrafung
    phase_sw = 1.0 if getattr(traffic_signal, "phase_changed", False) else 0.0

    # Reward-Berechnung
    r = -w_q * q_norm - w_build * b_norm + w_flow * f_norm - w_switch * phase_sw
    r = float(np.clip(r, -clip, clip))

    # Kleine positive Basis hinzufügen
    r += 0.1

    # Update interner Zustände
    traffic_signal._rw_state = {"prev_q": q, "ema_q": ema_q, "ema_f": ema_f}
    traffic_signal.phase_changed = False  # Reset Flag nach Verwendung

    return r

# ====== Reward-Funktion: Custom-Variante ([-1, 1] gebunden) ======
def custom_reward(traffic_signal):
    """
    Ausgeglichene Reward-Funktion:
    Bestraft Stau, Wartezeit, Teleports und Kollisionen,
    belohnt Durchfluss.
    Skaliert Werte mit tanh, um Extremwerte abzuflachen.
    """

    # Initialisierung von vorherigen Zuständen
    if not hasattr(traffic_signal, "_prev_queue"):
        traffic_signal._prev_queue = 0
    if not hasattr(traffic_signal, "_prev_wait_sum"):
        traffic_signal._prev_wait_sum = 0.0

    # Live-Metriken aus SUMO
    queue = traffic_signal.get_total_queued()
    wait_sum_total = float(np.sum(traffic_signal.get_accumulated_waiting_time_per_lane()))
    sim = traci.simulation
    arrived = sim.getArrivedNumber()
    teleports = sim.getStartingTeleportNumber()
    collisions = sim.getCollidingVehiclesNumber()

    # Änderungen pro Step (Delta)
    delta_queue = queue - traffic_signal._prev_queue
    delta_wait = max(0.0, wait_sum_total - traffic_signal._prev_wait_sum)

    # Update der gespeicherten Werte
    traffic_signal._prev_queue = queue
    traffic_signal._prev_wait_sum = wait_sum_total

    # Skaliertes, glattes Mapping via tanh()
    q_term       = -np.tanh(queue / 30.0)
    dq_term      = -np.tanh(max(0, delta_queue) / 10.0)
    wait_term    = -np.tanh(delta_wait / 60.0)
    arrived_term =  np.tanh(arrived / 5.0)
    tp_term      = -np.tanh(teleports / 1.0)
    col_term     = -np.tanh(collisions / 1.0)

    # Gewichtung der einzelnen Komponenten
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

    # Härtere Strafen für Teleports/Kollisionen
    if teleports > 0:
        raw -= 0.5
    if collisions > 0:
        raw -= 0.8

    # Ergebnis in [-1, 1] begrenzen
    reward = float(np.tanh(raw))
    return reward


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
            reward_fn="diff-waiting-time",
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
    """
    Loggt Episodenmetriken:
    - mean_*  → Episodenmittelwerte
    - total_* → Summen über die Episode
    """
    def __init__(self, prefix="episode", verbose=0):
        super().__init__(verbose)
        self.prefix = prefix
        self.verbose = verbose
        self.sums = {}
        self.counts = {}

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if infos is None:
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue
            for key, v in info.items():
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    continue

                if key.startswith("system_mean_"):
                    # Mittelwert → akkumulieren und später mitteln
                    self.sums[key] = self.sums.get(key, 0.0) + float(v)
                    self.counts[key] = self.counts.get(key, 0) + 1

                elif key.startswith("system_total_"):
                    # Totals → über Episode aufsummieren
                    self.sums[key] = self.sums.get(key, 0.0) + float(v)

        # Episode zu Ende?
        if dones is not None and any(dones):
            for k, total in self.sums.items():
                short_key = shorten_key(k)
                if k.startswith("system_mean_"):
                    mean_val = total / max(1, self.counts.get(k, 1))
                    self.logger.record(f"{self.prefix}/{short_key}", mean_val)
                    if self.verbose:
                        print(f"[EpisodeMetrics] {short_key} (mean) = {mean_val:.3f}")
                elif k.startswith("system_total_"):
                    self.logger.record(f"{self.prefix}/{short_key}", total)
                    if self.verbose:
                        print(f"[EpisodeMetrics] {short_key} (total) = {total:.0f}")

            # Reset für nächste Episode
            self.sums.clear()
            self.counts.clear()

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
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])), # Netzarchitektur
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
            total_timesteps=1_500_000,
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
