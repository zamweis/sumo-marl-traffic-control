import sys
import os
import datetime
from sumo_rl.environment.env import parallel_env

# Logs-Verzeichnis automatisch anlegen
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Zeitstempel für Dateinamen
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_dir, f"sumo_run_{timestamp}.log")

# TeeLogger – schreibt in Konsole und Datei gleichzeitig
class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.__stdout__  # Standardausgabe sichern
        self.log = open(log_path, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

# Logging starten
sys.stdout = TeeLogger(log_file_path)

# SUMO-RL Multiagent-Umgebung konfigurieren
env = parallel_env(
    net_file='synthetic.net.xml',
    route_file='synthetic.rou.xml',
    use_gui=True,
    num_seconds=600
)

try:
    obs = env.reset()
    step = 0
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        print(f"\n🔁 STEP {step}")
        for agent in env.agents:
            print(f"Agent: {agent}")
            print(f"   Action: {actions[agent]}")
            print(f"   Reward: {rewards[agent]}")
            print(f"   Observation: {obs[agent]}")
            print(f"   Terminated: {terminations[agent]}, Truncated: {truncations[agent]}")
            print(f"   Info: {infos[agent]}")
        step += 1

finally:
    print("\nSimulation ended. Closing environment.")
    env.close()
    sys.stdout.log.close()  # Nur Logdatei schließen
    sys.stdout = sys.stdout.terminal  # stdout zurück auf Konsole setzen
