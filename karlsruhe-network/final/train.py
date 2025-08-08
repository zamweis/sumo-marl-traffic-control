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
SEEDS = [1234, 3456, 5678, 7890]  # beliebig erweiterbar

# ==== Custom Reward Function ====
def custom_reward(traffic_signal):
    if not hasattr(traffic_signal, "prev_queue"):
        traffic_signal.prev_queue = traffic_signal.get_total_queued()

    queue = traffic_signal.get_total_queued()
    waiting = np.sum(traffic_signal.get_accumulated_waiting_time_per_lane())

    sim = traci.simulation
    arrived = sim.getArrivedNumber()
    teleport = sim.getStartingTeleportNumber()
    collisions = sim.getCollidingVehiclesNumber()

    delta_queue = traffic_signal.prev_queue - queue
    traffic_signal.prev_queue = queue

    reward = (
        -0.1 * queue
        - 0.05 * waiting
        - 2.0 * teleport
        - 10.0 * collisions
        + 1.0 * arrived
        + 0.3 * delta_queue
    )

    if teleport > 10 or collisions > 5:
        reward -= 20

    return np.clip(reward, -100, 100)

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
        num_seconds=1000,
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
            total_timesteps=1_500_000,
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
