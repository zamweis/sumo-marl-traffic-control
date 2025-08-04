from sumo_rl import SumoEnvironment
from sumolib import net
import traci
from xml.etree import ElementTree as ET

NET_FILE = "network.net.xml"
ROUTE_FILE = "routes.rou.xml"

print("Lade Netz...")
net = net.readNet(NET_FILE)

# TLS filtern
valid_tls_ids = []
valid_tls_ids = [tls.getID() for tls in net.getTrafficLights()]

print(f"Gültige TLS (nur 'traffic_light'): {len(valid_tls_ids)}")
print(f"TLS-IDs: {valid_tls_ids}")

# XML überprüfen
print("Prüfe Anzahl Phasen jeder TLS...")
tree = ET.parse(NET_FILE)
root = tree.getroot()
problematic_tls = []
for tls_id in valid_tls_ids:
    for logic in root.findall(f"./tlLogic[@id='{tls_id}']"):
        phases = logic.findall("phase")
        if len(phases) < 2:
            print(f"TLS {tls_id} hat nur {len(phases)} Phasen – kann Probleme verursachen.")
            problematic_tls.append(tls_id)

if problematic_tls:
    print(f"\nFehlerhafte TLS (weniger als 2 Phasen): {problematic_tls}")
    for tid in problematic_tls:
        valid_tls_ids.remove(tid)

print("Initialisiere SUMO-Umgebung im Multi-Agent-Modus...")
try:
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=True,
        single_agent=False,               # Multi-Agent aktivieren
        reward_fn="diff-waiting-time",
        delta_time=5,
        yellow_time=2,
        min_green=5,
        tls_ids=valid_tls_ids,           # TLS explizit übergeben
        fixed_ts=False
    )
except Exception as e:
    print("Fehler beim Initialisieren der Umgebung:")
    raise e

print("SUMO-Umgebung erfolgreich gestartet.")
print("▶Starte Test-Episode...")

# 🟢 Multi-Agent Schrittweise Simulation
obs = env.reset()
done = {"__all__": False}
step = 0

while not done["__all__"] and step < 1000:
    actions = {tls: env.action_space.sample() for tls in valid_tls_ids}
    obs, reward, done, _, _ = env.step(actions)
    print(f"Step {step}: Rewards = {reward}")
    step += 1

env.close()
print("Simulation beendet.")
