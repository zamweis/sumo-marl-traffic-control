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
from gym import Wrapper
import traci  # direkter Zugriff auf SUMO-Simulation

# ==== Custom Wrapper: Frühes Beenden, wenn keine Fahrzeuge mehr ====
class AutoTerminateWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # TraCI-Abfrage: keine Fahrzeuge mehr unterwegs oder geplant?
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("[AutoTerminate] Keine Fahrzeuge mehr → Episode terminiert.")
                done = {agent: True for agent in done}
        except Exception:
            pass  # z. B. falls TraCI noch nicht initialisiert wurde

        return obs, reward, done, info

# ==== Ordnerstruktur ====
run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("runs", f"ppo_sumo_{run_id}")
os.makedirs(log_dir, exist_ok=True)
model_path = os.path.join(log_dir, "model.zip")
normalize_path = os.path.join(log_dir, "vecnormalize.pkl")

# ==== SUMO-RL Multiagent-Umgebung ====
env = parallel_env(
    net_file="synthetic.net.xml",
    route_file="synthetic.rou.xml",
    use_gui=False,  # für Debug: False bei Batch-Training
    num_seconds=1000,
    reward_fn="queue",
    min_green=5,
    max_depart_delay=0,
)

# ==== Supersuit: Padding für Beobachtungen & Aktionen ====
env = pad_observations_v0(env)
env = pad_action_space_v0(env)

# ==== Vectorisierung für Stable-Baselines ====
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

# ==== Frühes Beenden, wenn Simulation leerläuft ====
env = AutoTerminateWrapper(env)

# ==== Monitoring & Reward-Normalisierung ====
env = VecMonitor(env)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# ==== PPO-Modell mit CPU ====
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

try:
    # ==== Training ====
    model.learn(
        total_timesteps=100_000,
        callback=checkpoint_callback,
    )

    # ==== Speichern ====
    model.save(model_path)
    env.save(normalize_path)  # Save VecNormalize-Statistiken
    print(f"\nTraining abgeschlossen. Modell gespeichert unter: {model_path}")
    print(f"VecNormalize gespeichert unter: {normalize_path}")

except Exception as e:
    print(f"\nFehler während Training: {e}")
finally:
    env.close()
