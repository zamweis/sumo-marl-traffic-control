import os
import shutil
import time
import datetime
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from supersuit import (
    pad_observations_v0,
    pad_action_space_v0,
    pettingzoo_env_to_vec_env_v1,
    concat_vec_envs_v1
)

from sumo_rl.environment.env import parallel_env

# === Setup ===
SEED = 42
NUM_ENVS = 2  # Anzahl paralleler Umgebungen
BASE_PORT = 8813
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Custom Reward ===
def custom_reward(traffic_signal):
    import traci
    queue = traffic_signal.get_total_queued()
    waiting = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    arrived = traci.simulation.getArrivedNumber()
    teleport = traci.simulation.getStartingTeleportNumber()
    collisions = traci.simulation.getCollidingVehiclesNumber()
    return -1.0 * queue - 0.1 * waiting - 5.0 * teleport - 20.0 * collisions + 0.5 * arrived

# === Temp Env Setup ===
def setup_temp_envs(num_envs):
    os.makedirs("temp_envs", exist_ok=True)
    for i in range(num_envs):
        env_dir = f"temp_envs/env_{i}"
        os.makedirs(env_dir, exist_ok=True)
        shutil.copy("synthetic.net.xml", os.path.join(env_dir, "synthetic.net.xml"))
        shutil.copy("synthetic.rou.xml", os.path.join(env_dir, "synthetic.rou.xml"))

# === Env Factory ===
def make_env_instance(index, base_port=BASE_PORT):
    port = base_port + index
    net = f"temp_envs/env_{index}/synthetic.net.xml"
    rou = f"temp_envs/env_{index}/synthetic.rou.xml"

    env = parallel_env(
        net_file=net,
        route_file=rou,
        use_gui=False,
        num_seconds=1000,
        reward_fn=custom_reward,
        min_green=0,
        max_depart_delay=0,
        port=port,
        additional_sumo_cmd="",  # <- WICHTIG: leer setzen!
    )

    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    return env

# === Callbacks ===
class TimeCheckpointCallback(BaseCallback):
    def __init__(self, interval_sec, save_path):
        super().__init__()
        self.interval_sec = interval_sec
        self.save_path = save_path
        self.last_time = time.time()

    def _on_step(self):
        if time.time() - self.last_time > self.interval_sec:
            path = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(path)
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.save_path, "vecnormalize.pkl"))
            print(f"[Checkpoint] Saved at {self.num_timesteps} steps")
            self.last_time = time.time()
        return True

# === Main ===
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    setup_temp_envs(NUM_ENVS)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("runs", f"ppo_parallel_{now}")
    os.makedirs(log_dir, exist_ok=True)

    base_env = make_env_instance(0)
    vec_env = concat_vec_envs_v1(base_env, NUM_ENVS, num_cpus=1, base_class="stable_baselines3")
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        batch_size=2048,
        n_steps=2048,
        learning_rate=lambda f: 1e-4 + f * (3e-4 - 1e-4),
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )

    callbacks = CallbackList([
        TimeCheckpointCallback(interval_sec=1800, save_path=log_dir)
    ])

    try:
        model.learn(total_timesteps=200_000, callback=callbacks)
        model.save(os.path.join(log_dir, "final_model.zip"))
        vec_env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    except Exception as e:
        print("Training error:", e)
    finally:
        vec_env.close()
