import xml.etree.ElementTree as ET

# === Konfiguration ===
net_file = "karlsruhe.net.xml"

# === Einlesen ===
tree = ET.parse(net_file)
root = tree.getroot()

# === Alle controlledLinks zählen ===
tls_controlled_links = {}
for connection in root.findall("connection"):
    if "tl" in connection.attrib and "linkIndex" in connection.attrib:
        tls_id = connection.attrib["tl"]
        tls_controlled_links.setdefault(tls_id, set()).add(int(connection.attrib["linkIndex"]))

# === Alle Phasen prüfen ===
def check_tls_lengths():
    print("Überprüfe alle TLS auf inkonsistente Phasenlängen...\n")
    any_issues = False
    for logic in root.findall("tlLogic"):
        tls_id = logic.attrib["id"]
        expected_len = len(tls_controlled_links.get(tls_id, []))

        if expected_len == 0:
            print(f" TLS '{tls_id}' hat keine controlledLinks (wird evtl. nicht gesteuert)")
            continue

        for i, phase in enumerate(logic.findall("phase")):
            actual_len = len(phase.attrib["state"])
            if actual_len != expected_len:
                print(f" Phase {i} von TLS '{tls_id}' hat Länge {actual_len}, erwartet: {expected_len}")
                print(f"    → state=\"{phase.attrib['state']}\"")
                any_issues = True

    if not any_issues:
        print(" Alle TLS-Phasen stimmen mit ihren controlledLinks überein!")

check_tls_lengths()
