import traci
import subprocess
import time

from sumo_rl.environment.env import SumoEnvironment

NET_FILE = "map_filtered.net.xml"
ROUTE_FILE = "map.rou.xml"

print("[INFO] Starte SUMO zur TLS-Prüfung...")

try:
    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,
        num_seconds=1,  # Wir wollen nur initiale TLS prüfen
    )
except Exception as e:
    print("[FATAL] SUMO-Fehler beim Initialisieren:")
    print(e)
    exit(1)

traci_conn = env.sumo  # direkter Zugriff auf TraCI-Verbindung

for tls_id in traci_conn.trafficlight.getIDList():
    try:
        links = traci_conn.trafficlight.getControlledLinks(tls_id)
        num_links = sum(len(l) for l in links)

        logic = traci_conn.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
        for i, phase in enumerate(logic.getPhases()):
            state_len = len(phase.state)
            if state_len != num_links:
                print(f"❌ TLS '{tls_id}' – Phase {i} hat {state_len} Zeichen (erwartet: {num_links})")
    except Exception as e:
        print(f"💥 Fehler bei TLS '{tls_id}': {e}")

print("[FERTIG] Prüfung abgeschlossen.")
env.close()
