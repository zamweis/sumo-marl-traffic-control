from sumo_rl import SumoEnvironment
from sumolib import net
import traci
from xml.etree import ElementTree as ET

NET_FILE = "karlsruhe3.net.xml"
ROUTE_FILE = "karlsruhe3.rou.xml"

print("🚦 Lade Netz...")
net = net.readNet(NET_FILE)

# TLS filtern
valid_tls_ids = []
for tls in net.getTrafficLights():
    junction = net.getNode(tls.getID())
    if junction.getType() == "traffic_light":
        valid_tls_ids.append(tls.getID())

print(f"🔍 Gültige TLS (nur 'traffic_light'): {len(valid_tls_ids)}")
print(f"🆔 TLS-IDs: {valid_tls_ids}")

# XML überprüfen
print("📋 Prüfe Anzahl Phasen jeder TLS...")
tree = ET.parse(NET_FILE)
root = tree.getroot()
problematic_tls = []
for tls_id in valid_tls_ids:
    for logic in root.findall(f"./tlLogic[@id='{tls_id}']"):
        phases = logic.findall("phase")
        if len(phases) < 2:
            print(f"⚠️ TLS {tls_id} hat nur {len(phases)} Phasen – kann Probleme verursachen.")
            problematic_tls.append(tls_id)

if problematic_tls:
    print(f"\n❌ Fehlerhafte TLS (weniger als 2 Phasen): {problematic_tls}")
    for tid in problematic_tls:
        valid_tls_ids.remove(tid)

print("🧠 Initialisiere SUMO-Umgebung...")
try:
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=True,
        single_agent=True,
        reward_fn="diff-waiting-time",
        delta_time=5,
        yellow_time=2,
        min_green=5,
        fixed_ts=True
    )
except Exception as e:
    print("❌ Fehler beim Initialisieren der Umgebung:")
    raise e

print("✅ SUMO-Umgebung erfolgreich gestartet.")
print("▶️ Starte Test-Episode...")

# 🟢 Hier wird die Simulation gestartet
obs = env.reset()
done = False
step = 0

while not done and step < 20:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Step {step}: Reward = {reward}")
    step += 1

env.close()
print("🏁 Simulation beendet.")
