import os
import re
import time
import datetime
import traci
import numpy as np
import torch
import json
import optuna
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

SEED = 999
np.random.seed(SEED)
torch.manual_seed(SEED)

def custom_reward(traffic_signal):
    # Aktuelle Werte abfragen
    queue = traffic_signal.get_total_queued()
    waiting = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    arrived = traci.simulation.getArrivedNumber()
    teleport = traci.simulation.getStartingTeleportNumber()
    collisions = traci.simulation.getCollidingVehiclesNumber()

    # Delta-Warteschlange: Verbesserung belohnen
    prev_queue = getattr(traffic_signal, "prev_queue", queue)
    delta_queue = prev_queue - queue
    traffic_signal.prev_queue = queue

    # Reward-Zusammensetzung (angepasste Gewichtungen)
    reward = (
        (-0.1 * queue)                     # Leichte Bestrafung für Staus
        + (-0.05 * waiting)                # Leichte Bestrafung für Wartezeit
        + (-2.0 * teleport)                # Mäßige Bestrafung für Teleports
        + (-10.0 * collisions)             # Mäßige Bestrafung für Kollisionen
        + (1.0 * arrived)                  # Gute Belohnung für ankommende Fahrzeuge
        + (0.3 * delta_queue)              # Belohnung für Reduktion von Warteschlangen
    )

    # Sanfte, pauschale Bestrafung bei Netz-Kollaps (aber nur einmalig)
    if teleport > 10 or collisions > 5:
        reward -= 20

    # Reward clipping für Sicherheit
    reward = np.clip(reward, -100, 100)

    # Optional: Debug-Log (auskommentieren für produktive Runs)
    # print(f"Reward breakdown: queue={queue}, waiting={waiting}, arrived={arrived}, teleport={teleport}, collisions={collisions}, delta_queue={delta_queue}, reward={reward}")

    return reward

def linear_schedule(start):
    return lambda progress: start * (1 - progress)

def dynamic_clip_range(start=0.2):
    return lambda progress: max(0.1, start * (1 - 0.5 * progress))

def adaptive_entropy_schedule(start=0.01):
    return lambda progress: max(0.001, start * (1 - progress))

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

        if isinstance(self.model.ent_coef, float):
            self.logger.record("train/ent_coef", self.model.ent_coef)
        else:
            self.logger.record("train/ent_coef", self.model.ent_coef(progress))

        return True

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

def make_env():
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
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=8, base_class="stable_baselines3")
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env

def objective(trial):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", f"ppo_sumo_optuna_{now}")
    os.makedirs(log_dir, exist_ok=True)

    env = make_env()

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.02)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    net_arch = trial.suggest_categorical("net_arch", [[64, 64], [128, 128], [256, 256]])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        tensorboard_log=log_dir,
        learning_rate=linear_schedule(learning_rate),
        clip_range=dynamic_clip_range(clip_range),
        ent_coef=adaptive_entropy_schedule(ent_coef),
        batch_size=1024,
        n_steps=1024,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=net_arch, vf=net_arch)),
    )

    model.learn(total_timesteps=50000)

    rewards = [ep_info["r"] for ep_info in model.ep_info_buffer if "r" in ep_info]
    mean_reward = np.mean(rewards) if rewards else -np.inf

    env.close()
    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Beste Parameterkombination:")
    print(study.best_params)
