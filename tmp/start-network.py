from sumo_rl import SumoEnvironment
from sumolib import net
import traci
from xml.etree import ElementTree as ET

NET_FILE = "network2.net.xml"
ROUTE_FILE = "network2.rou.xml"

print("🚦 Lade Netz...")
net = net.readNet(NET_FILE)

# TLS-IDs direkt übernehmen
valid_tls_ids = [tls.getID() for tls in net.getTrafficLights()]
print(f"🔍 Gefundene TLS: {len(valid_tls_ids)}")
print(f"🆔 TLS-IDs: {valid_tls_ids}")

# XML prüfen (Phasenanzahl)
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

# Filtere problematische TLS heraus
valid_tls_ids = [tid for tid in valid_tls_ids if tid not in problematic_tls]

print(f"✅ Verbleibende gültige TLS: {len(valid_tls_ids)}")

# Starte SUMO-Umgebung
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

obs = env.reset()
done = False
step = 0

while not done and step < 5000:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Step {step}: Reward = {reward}")
    step += 1

env.close()
print("🏁 Simulation beendet.")
