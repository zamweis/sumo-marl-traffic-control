import os
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from supersuit import (
    pad_observations_v0,
    pad_action_space_v0,
    pettingzoo_env_to_vec_env_v1,
    concat_vec_envs_v1
)
from sumo_rl.environment.env import parallel_env
from sumo_rl.environment.traffic_signal import TrafficSignal
from gym import Wrapper
import traci

# ==== Custom Reward Function ====
def custom_reward(traffic_signal):
    queue = traffic_signal.get_total_queued()
    waiting = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    departed = traffic_signal.env.sumo.simulation.getArrivedNumber()
    return -1.0 * queue - 0.1 * waiting + 0.5 * departed

# === Registrierung MUSS VOR env-Aufruf geschehen ===
TrafficSignal.reward_fns["custom"] = custom_reward

# ==== AutoTerminateWrapper: Frühes Beenden ====
class AutoTerminateWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("[AutoTerminate] Keine Fahrzeuge mehr → Episode terminiert.")
                done = {agent: True for agent in done}
        except Exception:
            pass
        return obs, reward, done, info

# ==== Ordnerstruktur ====
run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("runs", f"ppo_sumo_{run_id}")
os.makedirs(log_dir, exist_ok=True)
model_path = os.path.join(log_dir, "model.zip")
normalize_path = os.path.join(log_dir, "vecnormalize.pkl")

# ==== SUMO-RL Multiagent-Umgebung mit custom Reward ====
env = parallel_env(
    net_file="synthetic.net.xml",
    route_file="synthetic.rou.xml",
    use_gui=False,
    num_seconds=1000,
    reward_fn="custom",  # <- Jetzt korrekt referenziert
    min_green=5,
    max_depart_delay=0,
)

# ==== Wrapping ====
env = pad_observations_v0(env)
env = pad_action_space_v0(env)
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
env = AutoTerminateWrapper(env)
env = VecMonitor(env)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# ==== PPO-Modell ====
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=log_dir,
    batch_size=2048,
    n_steps=2048,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    device="cpu"
)

# ==== Checkpoints ====
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=log_dir,
    name_prefix="ppo_sumo_model",
    verbose=1,
)

# ==== Training ====
try:
    model.learn(
        total_timesteps=100_000,
        callback=checkpoint_callback,
    )
    model.save(model_path)
    env.save(normalize_path)
    print(f"\nTraining abgeschlossen. Modell gespeichert unter: {model_path}")
    print(f"VecNormalize gespeichert unter: {normalize_path}")
except Exception as e:
    print(f"\nFehler während Training: {e}")
finally:
    env.close()
