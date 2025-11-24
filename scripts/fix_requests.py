import xml.etree.ElementTree as ET

net_file = "karlsruhe.net.xml"
output_file = "karlsruhe_fixed_tls.net.xml"

tree = ET.parse(net_file)
root = tree.getroot()

# Finde maximal verwendete Signal-Indices pro TLS
tls_max_index = {}
for conn in root.findall("connection"):
    tl = conn.get("tl")
    idx = conn.get("linkIndex")
    if tl and idx:
        idx = int(idx)
        tls_max_index[tl] = max(tls_max_index.get(tl, -1), idx)

# Bereinigung
total_removed_requests = 0
total_adjusted_phases = 0
changed_tls = []

for junction in root.findall("junction"):
    tls_id = junction.get("id")
    if tls_id not in tls_max_index:
        continue

    max_idx = tls_max_index[tls_id]
    requests = list(junction.findall("request"))
    removed = 0

    for req in requests:
        req_idx = int(req.get("index"))
        if req_idx > max_idx:
            junction.remove(req)
            removed += 1

    if removed > 0:
        print(f"🧹 TLS '{tls_id}': {removed} ungültige <request>-Einträge entfernt.")
        total_removed_requests += removed
        changed_tls.append(tls_id)

    # Kürze zugehörige Phasen
    for tl in root.findall("tlLogic"):
        if tl.get("id") == tls_id:
            adjusted = 0
            for phase in tl.findall("phase"):
                state = phase.get("state")
                if len(state) > max_idx + 1:
                    old_len = len(state)
                    phase.set("state", state[:max_idx + 1])
                    adjusted += 1
            if adjusted > 0:
                print(f" TLS '{tls_id}': {adjusted} <phase>-Strings auf Länge {max_idx + 1} gekürzt.")
                total_adjusted_phases += adjusted
                if tls_id not in changed_tls:
                    changed_tls.append(tls_id)

# Speichern
tree.write(output_file, encoding="utf-8")
print("\n Reparatur abgeschlossen.")
print(f" Gesamt entfernte <request>-Einträge: {total_removed_requests}")
print(f" Gesamt angepasste <phase>-Einträge: {total_adjusted_phases}")
print(f" Betroffene TLS-IDs: {len(changed_tls)} Stück")
for tls in changed_tls:
    print(f"  - {tls}")
print(f"\n Bereinigte Datei gespeichert unter: {output_file}")
