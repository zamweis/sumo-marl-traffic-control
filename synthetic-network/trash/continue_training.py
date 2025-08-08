import os
import datetime
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_rl.environment.env import parallel_env
from supersuit import (
    pad_observations_v0,
    pad_action_space_v0,
    pettingzoo_env_to_vec_env_v1,
    concat_vec_envs_v1
)
from gym import Wrapper

# ==== Custom Wrapper: Frühes Beenden, wenn keine Fahrzeuge mehr ====
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

# ==== Suche nach dem neuesten vollständigen Run ====
def find_latest_complete_run(base_dir="runs", prefix="ppo_sumo_"):
    subdirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith(prefix)],
        reverse=True
    )
    for d in subdirs:
        model_path = os.path.join(base_dir, d, "model.zip")
        norm_path = os.path.join(base_dir, d, "vecnormalize.pkl")
        if os.path.exists(model_path) and os.path.exists(norm_path):
            return os.path.join(base_dir, d), model_path, norm_path
    raise FileNotFoundError("Kein vollständiger Run mit model.zip und vecnormalize.pkl gefunden.")

# ==== Neues Log-Verzeichnis (gleiches Format wie Initialtraining) ====
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("runs", f"ppo_sumo_{now}")
os.makedirs(log_dir, exist_ok=True)

# ==== Lade letzten vollständigen Run ====
latest_run_dir, model_path, normalize_path = find_latest_complete_run()

print("Fortsetzung wird gestartet mit:")
print(f"Verzeichnis : {latest_run_dir}")
print(f"Modell      : {model_path}")
print(f"Normalize   : {normalize_path}\n")

# ==== SUMO-RL Umgebung ====
env = parallel_env(
    net_file="synthetic.net.xml",
    route_file="synthetic.rou.xml",
    use_gui=True,
    num_seconds=1000,
    reward_fn="queue",
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
env = VecNormalize.load(normalize_path, env)
env.training = True
env.norm_reward = True

# ==== PPO laden ====
model = PPO.load(model_path, env=env, tensorboard_log=log_dir, verbose=1)

# ==== Checkpoints ====
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=log_dir,
    name_prefix="ppo_sumo_model",
    verbose=1,
)

# ==== Weitertrainieren ====
try:
    model.learn(
        total_timesteps=500_000,
        callback=checkpoint_callback,
    )

    # ==== Speichern ====
    model.save(os.path.join(log_dir, "model.zip"))
    env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print(f"\nTraining abgeschlossen. Modell gespeichert unter: {log_dir}")

except Exception as e:
    print(f"\nFehler während Training: {e}")
finally:
    env.close()
