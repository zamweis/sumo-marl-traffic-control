import traci
import re

sumo_binary = "sumo"
net_file = "network.net.xml"
route_file = "routes.rou.xml"

sumo_cmd = [sumo_binary, "-n", net_file, "-r", route_file, "--start", "--time-to-teleport", "-1"]

print("Starte SUMO...")
traci.start(sumo_cmd)

tls_ids = traci.trafficlight.getIDList()
print(f"Anzahl erkannter TLS: {len(tls_ids)}")

invalid_tls = []

for tls_id in tls_ids:
    print(f"\n Prüfe TLS: {tls_id}")
    try:
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        if not logics:
            print("Kein Steuerprogramm vorhanden.")
            invalid_tls.append((tls_id, "Keine Logik"))
            continue

        logic = logics[0]
        phases = logic.phases
        print(f"Phasen: {len(phases)}")

        if len(phases) < 2:
            print("Weniger als 2 Phasen → ungeeignet für SUMO-RL")
            invalid_tls.append((tls_id, "Zu wenige Phasen"))
        else:
            for i, phase in enumerate(phases):
                state = phase.state
                duration = phase.duration
                print(f"    - Phase {i}: Dauer={duration}s, State={state}")
                if duration <= 0:
                    print(f"    Ungültige Dauer in Phase {i}")
                    invalid_tls.append((tls_id, "Phase mit Dauer <= 0"))
                if not re.fullmatch(r"[rygGuR]*", state):
                    print(f"    Ungültige Zeichen in Phase {i}: {state}")
                    invalid_tls.append((tls_id, f"Ungültige Phase {i}"))

        links = traci.trafficlight.getControlledLinks(tls_id)
        print(f"  ➤ Gesteuerte Links: {len(links)}")
        if len(links) == 0:
            print("  Keine gesteuerten Links")
            invalid_tls.append((tls_id, "Keine gesteuerten Verbindungen"))

    except Exception as e:
        print(f" Fehler beim Prüfen von TLS {tls_id}: {e}")
        invalid_tls.append((tls_id, f"Traci-Fehler: {e}"))

traci.close()

print("\nZusammenfassung:")
if not invalid_tls:
    print("Alle TLS sehen funktionsfähig aus.")
else:
    for tls_id, reason in invalid_tls:
        print(f"  - {tls_id}: {reason}")
