import os
import re
import time
import datetime
import traci
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

# ==== Globaler Speicher für SUMO-Metriken pro Schritt ====
GLOBAL_SIM_METRICS = {
    "teleport": 0.0,
    "collisions": 0.0,
    "arrived": 0.0,
}

# ==== Reward-Funktion (agent-spezifisch) ====
class RewardFunction:
    def __init__(self, weights=None):
        self.weights = weights or {
            "queue": -1.0,
            "waiting": -0.1,
            "teleport": -5.0,
            "collisions": -20.0,
            "arrived": 0.5,
        }
        self.agent_data = {}

    def compute(self, traffic_signal):
        tls_id = traffic_signal.id
        if tls_id not in self.agent_data:
            self.agent_data[tls_id] = {"queue": 0.0, "waiting": 0.0}

        try:
            queue = traffic_signal.get_total_queued()
            waiting = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
        except Exception as e:
            print(f"[WARN] Fehler bei Agent {tls_id}: {e}")
            queue = waiting = 0.0

        self.agent_data[tls_id]["queue"] += queue
        self.agent_data[tls_id]["waiting"] += waiting

        return self.weights["queue"] * queue + self.weights["waiting"] * waiting

    def get_aggregated_metrics(self):
        result = {"queue": 0.0, "waiting": 0.0}
        for data in self.agent_data.values():
            for k in result:
                result[k] += data.get(k, 0.0)
        result.update(GLOBAL_SIM_METRICS)
        return result

    def reset(self):
        self.agent_data = {}
        for k in GLOBAL_SIM_METRICS:
            GLOBAL_SIM_METRICS[k] = 0.0

# ==== Callback für SUMO-Metriken pro Schritt ====
class GlobalMetricCollector(BaseCallback):
    def _on_step(self) -> bool:
        try:
            GLOBAL_SIM_METRICS["arrived"] += traci.simulation.getArrivedNumber()
            GLOBAL_SIM_METRICS["teleport"] += traci.simulation.getStartingTeleportNumber()
            GLOBAL_SIM_METRICS["collisions"] += traci.simulation.getCollidingVehiclesNumber()
        except Exception as e:
            print(f"[WARN] Fehler bei globalen Metriken: {e}")
        return True

# ==== Callback für Rollout-Debug ====
class RewardDebugCallback(BaseCallback):
    def __init__(self, reward_fn_instance, interval_rollouts=1, verbose=0):
        super().__init__(verbose)
        self.reward_fn = reward_fn_instance
        self.interval_rollouts = interval_rollouts
        self.rollout_count = 0

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        if self.rollout_count % self.interval_rollouts != 0:
            self.reward_fn.reset()
            return

        data = self.reward_fn.get_aggregated_metrics()
        total_reward = 0.0

        print("\n[Reward Debug – Rollout-Aggregat]")
        for k, w in self.reward_fn.weights.items():
            v = data.get(k, 0.0)
            print(f"  {k:<10}: {v:.2f} × {w:<5} = {v * w:.2f}")
            total_reward += v * w
        print(f"  → Totaler Rollout-Reward: {total_reward:.2f}")
        print("-" * 42)

        self.reward_fn.reset()

    def _on_step(self) -> bool:
        return True

# ==== Frühes Beenden wenn keine Fahrzeuge ====
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

# ==== Logging-Verzeichnis ====
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("runs", f"ppo_sumo_{now}")
os.makedirs(log_dir, exist_ok=True)

# ==== Reward-Funktion-Instanz ====
reward_fn_instance = RewardFunction()

# ==== SUMO-RL Umgebung ====
env = parallel_env(
    net_file="synthetic.net.xml",
    route_file="synthetic.rou.xml",
    use_gui=False,
    num_seconds=1000,
    reward_fn=lambda ts: reward_fn_instance.compute(ts),
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

# ==== Modell laden oder neu erstellen ====
result = find_latest_complete_run()
if result:
    latest_run_dir, model_path, normalize_path = result
    print("Fortsetzung wird gestartet mit:")
    print(f"Verzeichnis : {latest_run_dir}")
    print(f"Modell      : {model_path}")
    print(f"Normalize   : {normalize_path}\n")

    env = VecNormalize.load(normalize_path, env)
    env.training = True
    env.norm_reward = True

    model = PPO.load(model_path, env=env, tensorboard_log=log_dir, verbose=1)
    print(f"[INFO] Modell startet bei {model.num_timesteps} Timesteps.")
else:
    print("[INFO] Kein vorheriges Modell gefunden. Starte frisches Training.\n")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        batch_size=2048,
        n_steps=2048,
        learning_rate=lambda progress: progress * 3e-4,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu"
    )

# ==== Zeitbasierter Checkpoint Callback ====
class TimeBasedCheckpointCallback(BaseCallback):
    def __init__(self, save_interval_sec, save_path, name_prefix="ppo_sumo_model", verbose=0):
        super().__init__(verbose)
        self.save_interval_sec = save_interval_sec
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_save_time = time.time()

    def _on_step(self) -> bool:
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval_sec:
            timestep = self.num_timesteps
            filename = f"{self.name_prefix}_{timestep}_steps"
            self.model.save(os.path.join(self.save_path, filename + ".zip"))
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.save_path, "vecnormalize.pkl"))
            print(f"[Zeit-Checkpoint] Modell gespeichert bei {timestep} Schritten ({filename})")
            self.last_save_time = current_time
        return True

# ==== Callbacks registrieren ====
callbacks = CallbackList([
    TimeBasedCheckpointCallback(
        save_interval_sec=1800,
        save_path=log_dir,
        name_prefix="ppo_sumo_model",
        verbose=1,
    ),
    GlobalMetricCollector(),
    RewardDebugCallback(
        reward_fn_instance=reward_fn_instance,
        interval_rollouts=1
    ),
])

# ==== Training starten ====
try:
    model.learn(
        total_timesteps=1_000_000,
        callback=callbacks,
    )
    model.save(os.path.join(log_dir, "model.zip"))
    env.save(os.path.join(log_dir, "vecnormalize.pkl"))
    print(f"\nTraining abgeschlossen. Modell gespeichert unter: {log_dir}")

except Exception as e:
    print(f"\nFehler während Training: {e}")

finally:
    env.close()