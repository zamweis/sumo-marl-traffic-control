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

# ==== Custom Reward Function ====
def custom_reward(traffic_signal):
    queue = traffic_signal.get_total_queued()
    waiting = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    departed = traffic_signal.env.sumo.simulation.getArrivedNumber()
    return -1.0 * queue - 0.1 * waiting + 0.5 * departed

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

        # 1. Suche nach finalem Modell
        final_model = os.path.join(dir_path, "model.zip")
        if os.path.exists(final_model):
            return dir_path, final_model, norm_path

        # 2. Suche nach bestem Checkpoint-Modell mit Schrittzahl
        checkpoint_models = [
            f for f in os.listdir(dir_path)
            if re.match(r"ppo_sumo_model_(\d+)_steps\.zip", f)
        ]
        if checkpoint_models:
            checkpoint_models.sort(key=lambda x: int(re.findall(r"\d+", x)[0]), reverse=True)
            best_checkpoint = checkpoint_models[0]
            return dir_path, os.path.join(dir_path, best_checkpoint), norm_path

    return None  # Nichts gefunden

# ==== Logging ====
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("runs", f"ppo_sumo_{now}")
os.makedirs(log_dir, exist_ok=True)

# ==== SUMO-RL Umgebung ====
env = parallel_env(
    net_file="synthetic.net.xml",
    route_file="synthetic.rou.xml",
    use_gui=False,
    num_seconds=1000,
    reward_fn=custom_reward,
    min_green=5,
    max_depart_delay=0,
)

# ==== Wrapping ====
env = pad_observations_v0(env)
env = pad_action_space_v0(env)
env = pettingzoo_env_to_vec_env_v1(env)
env = concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
#env = AutoTerminateWrapper(env)
env = VecMonitor(env)

# ==== Laden oder neu initialisieren ====
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
        learning_rate=3e-4,
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

checkpoint_callback = TimeBasedCheckpointCallback(
    save_interval_sec=3600,  # 60 Minuten
    save_path=log_dir,
    name_prefix="ppo_sumo_model",
    verbose=1,
)

callbacks = CallbackList([checkpoint_callback])

# ==== Weitertrainieren ====
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
    # ==== Evaluation nach dem Training ====
    print("\n=== Beginne Evaluation des trainierten Modells ===")

    eval_env = parallel_env(
        net_file="synthetic.net.xml",
        route_file="synthetic.rou.xml",
        use_gui=False,
        num_seconds=1000,
        reward_fn=custom_reward,
        min_green=5,
        max_depart_delay=0,
    )
    eval_env = pad_observations_v0(eval_env)
    eval_env = pad_action_space_v0(eval_env)
    eval_env = pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = concat_vec_envs_v1(eval_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    eval_env = AutoTerminateWrapper(eval_env)
    eval_env = VecMonitor(eval_env)

    norm_path = os.path.join(log_dir, "vecnormalize.pkl")
    if os.path.exists(norm_path):
        eval_env = VecNormalize.load(norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        print(f"[WARNUNG] Normalisierungsdaten ({norm_path}) nicht gefunden. Evaluation könnte abweichen.")

    # Evaluation-Parameter
    eval_episodes = 5
    total_rewards = []
    total_arrived = []
    total_vehicle_counts = []
    total_avg_speeds = []
    total_stopped = []

    for ep in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0
        ep_arrived = 0
        ep_vehicle_count = 0
        ep_speed_sum = 0
        ep_speed_count = 0
        ep_stopped = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

            ep_reward += reward[0] if isinstance(reward, dict) else reward

            try:
                veh_ids = traci.vehicle.getIDList()
                ep_vehicle_count = max(ep_vehicle_count, len(veh_ids))

                # Geschwindigkeit aller Fahrzeuge aufsummieren
                for vid in veh_ids:
                    ep_speed_sum += traci.vehicle.getSpeed(vid)
                    ep_speed_count += 1

                # Gestoppte Fahrzeuge aufsummieren
                lane_ids = traci.lane.getIDList()
                for lid in lane_ids:
                    ep_stopped += traci.lane.getLastStepHaltingNumber(lid)

            except:
                pass

            if all(done.values()) if isinstance(done, dict) else done:
                try:
                    ep_arrived = traci.simulation.getArrivedNumber()
                except:
                    ep_arrived = -1
                break

        total_rewards.append(ep_reward)
        total_arrived.append(ep_arrived)
        total_vehicle_counts.append(ep_vehicle_count)
        total_stopped.append(ep_stopped)
        avg_speed = ep_speed_sum / ep_speed_count if ep_speed_count > 0 else 0
        total_avg_speeds.append(avg_speed)

        print(f"[Eval] Episode {ep+1}: Reward={ep_reward:.2f}, Arrived={ep_arrived}, Max Vehicles={ep_vehicle_count}, Ø Speed={avg_speed:.2f}, Stopped={ep_stopped}")

    avg_reward = sum(total_rewards) / eval_episodes
    avg_arrived = sum(total_arrived) / eval_episodes if all(a >= 0 for a in total_arrived) else "N/A"
    avg_veh = sum(total_vehicle_counts) / eval_episodes
    avg_speed_overall = sum(total_avg_speeds) / eval_episodes
    avg_stopped = sum(total_stopped) / eval_episodes

    print("\n=== Evaluation abgeschlossen ===")
    print(f"Durchschnittlicher Reward:        {avg_reward:.2f}")
    print(f"Ø angekommene Fahrzeuge:          {avg_arrived}")
    print(f"Ø max Fahrzeuge im Netz:          {avg_veh:.2f}")
    print(f"Ø Geschwindigkeit (km/h):         {avg_speed_overall:.2f}")
    print(f"Ø gestoppte Fahrzeuge pro Episode:{avg_stopped:.2f}")
    print("==============================\n")

    # CSV speichern
    import csv
    csv_path = os.path.join(log_dir, "evaluation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Arrived", "MaxVehicles", "AvgSpeed", "Stopped"])
        for i in range(eval_episodes):
            writer.writerow([
                i + 1,
                total_rewards[i],
                total_arrived[i],
                total_vehicle_counts[i],
                total_avg_speeds[i],
                total_stopped[i]
            ])
        writer.writerow(["Ø", avg_reward, avg_arrived, avg_veh, avg_speed_overall, avg_stopped])
    print(f"[INFO] Evaluationsergebnisse gespeichert unter {csv_path}")

    eval_env.close()
