import os
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from sumo_rl.environment.env import parallel_env

# ==== Ordnerstruktur ====
run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("runs", f"ppo_sumo_{run_id}")
os.makedirs(log_dir, exist_ok=True)
model_path = os.path.join(log_dir, "model.zip")

# ==== SUMO-RL Multiagent-Umgebung ====
env = parallel_env(
    net_file="synthetic.net.xml",
    route_file="synthetic.rou.xml",
    use_gui=False,               # Kein GUI = schneller
    num_seconds=600,             # Simulationsdauer pro Episode
    reward_fn="queue",           # Standard: Wartezeit minimieren
    min_green=5,
    max_depart_delay=0,
)

# ==== Environment-Wrapper für Stable-Baselines ====
# Vectorize PettingZoo env (multi-agent → single-agent vectorized)
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class=DummyVecEnv)
env = VecMonitor(env)

# ==== PPO-Modell initialisieren ====
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
)

# ==== Checkpoints ====
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=log_dir,
    name_prefix="ppo_sumo_model",
    verbose=1,
)

# ==== Training starten ====
model.learn(
    total_timesteps=100_000,
    callback=checkpoint_callback,
)

# ==== Modell speichern ====
model.save(model_path)
print(f"\nTraining abgeschlossen. Modell gespeichert unter: {model_path}")
