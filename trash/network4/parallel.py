from sumo_rl import SumoEnvironment
from sumolib import net
from xml.etree import ElementTree as ET
import os

# 🔧 Dateien setzen – du kannst hier absolute oder relative Pfade verwenden
NET_FILE = "network.net.xml"
ROUTE_FILE = "routes.rou.xml"

# Optional: Existenz prüfen
assert os.path.exists(NET_FILE), f"Datei {NET_FILE} nicht gefunden!"
assert os.path.exists(ROUTE_FILE), f"Datei {ROUTE_FILE} nicht gefunden!"

print("🚦 Lade Netzwerk und extrahiere TLS...")
net_data = net.readNet(NET_FILE)
tls_ids = [tls.getID() for tls in net_data.getTrafficLights()]

# Optional: Filterung von TLS mit < 2 Phasen
print("📋 Prüfe Anzahl Phasen jeder TLS...")
tree = ET.parse(NET_FILE)
root = tree.getroot()
filtered_tls_ids = []
for tls_id in tls_ids:
    for logic in root.findall(f"./tlLogic[@id='{tls_id}']"):
        phases = logic.findall("phase")
        if len(phases) >= 2:
            filtered_tls_ids.append(tls_id)
        else:
            print(f"⚠️ TLS {tls_id} hat nur {len(phases)} Phasen – wird übersprungen.")

print(f"✅ TLS-IDs für Training: {filtered_tls_ids}")

print("🧠 Initialisiere SUMO-Umgebung (Multi-Agent)...")
env = SumoEnvironment(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    use_gui=True,
    single_agent=False,
    reward_fn="diff-waiting-time",
    delta_time=5,
    yellow_time=2,
    min_green=5,
    fixed_ts=False
    # ❗ KEIN tls_ids=... notwendig (wird automatisch übernommen)
)

print("▶️ Starte Test-Episode...")
obs = env.reset()
done = {"__all__": False}
step = 0

# Zugriff auf TLS-IDs direkt über env.agents (das sind Agent-IDs = TLS)
while not done["__all__"] and step < 1000:
    actions = {agent: env.action_space.sample() for agent in env.agents}
    obs, rewards, done, _, _ = env.step(actions)
    print(f"Step {step} - Rewards: {rewards}")
    step += 1

env.close()
print("🏁 Simulation beendet.")
