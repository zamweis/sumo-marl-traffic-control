# ==== Imports ====
import os
import time
import datetime
import traci
import numpy as np
import csv
from gymnasium import Env, Wrapper, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl.environment.env import parallel_env

# ==== Custom Reward Function ====
def custom_reward(traffic_signal):
    queue = traffic_signal.get_total_queued()
    waiting = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    arrived = traci.simulation.getArrivedNumber()
    teleport = traci.simulation.getStartingTeleportNumber()
    collisions = traci.simulation.getCollidingVehiclesNumber()
    reward = (
        -1.0 * queue
        -0.1 * waiting
        -5.0 * teleport
        -20.0 * collisions
        +0.5 * arrived
    )
    return reward

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

# ==== Logging Setup ====
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("runs", f"ppo_sumo_multi_{now}")
os.makedirs(log_dir, exist_ok=True)

# ==== Multi-Agent Training Setup ====
agents_env = parallel_env(
    net_file="synthetic.net.xml",
    route_file="synthetic.rou.xml",
    use_gui=False,
    num_seconds=1000,
    reward_fn=custom_reward,
    min_green=5,
    max_depart_delay=0,
)
agents = agents_env.possible_agents

# ==== SingleAgentEnv für gymnasium-kompatibles SB3 ====
class SingleAgentEnv(Env):
    def __init__(self, agent_id):
        super().__init__()
        self.agent_id = agent_id
        self.env = parallel_env(
            net_file="synthetic.net.xml",
            route_file="synthetic.rou.xml",
            use_gui=False,
            num_seconds=1000,
            reward_fn=custom_reward,
            min_green=5,
            max_depart_delay=0,
        )
        self.observation_space = self.env.observation_space(self.agent_id)
        self.action_space = self.env.action_space(self.agent_id)
        self._agent_done = False

    def reset(self, *, seed=None, options=None):
        obs_dict, info = self.env.reset(seed=seed)
        self._agent_done = False
        return obs_dict[self.agent_id], info.get(self.agent_id, {})

    def step(self, action):
        if self._agent_done:
            raise RuntimeError("step() called after done=True")
        actions = {self.agent_id: action}
        obs, reward, done, info = self.env.step(actions)
        self._agent_done = done[self.agent_id]
        return obs[self.agent_id], reward[self.agent_id], done[self.agent_id], False, info[self.agent_id]

    def render(self):
        pass

    def close(self):
        self.env.close()

# ==== Callback für Checkpoints ====
class TimeCheckpointCallback(BaseCallback):
    def __init__(self, save_interval_sec, save_path, agent_name):
        super().__init__()
        self.save_interval_sec = save_interval_sec
        self.save_path = save_path
        self.agent_name = agent_name
        self.last_save_time = time.time()

    def _on_step(self) -> bool:
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval_sec:
            model_path = os.path.join(self.save_path, f"{self.agent_name}_checkpoint.zip")
            self.model.save(model_path)
            print(f"[Checkpoint] {self.agent_name} gespeichert: {model_path}")
            self.last_save_time = current_time
        return True

# ==== Training ====
models = {}
for agent in agents:
    print(f"\n[INFO] Starte Training für {agent}")
    agent_dir = os.path.join(log_dir, agent)
    os.makedirs(agent_dir, exist_ok=True)
    env = SingleAgentEnv(agent)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=agent_dir,
        batch_size=1024,
        n_steps=1024,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu"
    )
    callback = TimeCheckpointCallback(save_interval_sec=1800, save_path=agent_dir, agent_name=agent)
    model.learn(total_timesteps=500_000, callback=callback)
    model.save(os.path.join(agent_dir, "final_model.zip"))
    env.close()
    print(f"[INFO] Training für {agent} abgeschlossen und gespeichert.")
    models[agent] = model

print("\n[INFO] Alle Agenten wurden separat trainiert.")
