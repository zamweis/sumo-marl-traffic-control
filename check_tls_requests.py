import xml.etree.ElementTree as ET

net_file = "karlsruhe.net.xml"
tree = ET.parse(net_file)
root = tree.getroot()

# Zähle für jedes TLS wie viele signal indices es gibt (controlled links)
tls_signal_indices = {}
for conn in root.findall("connection"):
    if "tl" in conn.attrib and "linkIndex" in conn.attrib:
        tls_id = conn.attrib["tl"]
        tls_signal_indices.setdefault(tls_id, set()).add(int(conn.attrib["linkIndex"]))

# Vergleiche mit den request-Elementen
print("Überprüfe request-Indizes gegen Signalindizes...\n")
any_issues = False
for junction in root.findall("junction"):
    tls_id = junction.attrib.get("id")
    requests = junction.findall("request")
    if tls_id in tls_signal_indices:
        expected_max = len(tls_signal_indices[tls_id])
        for req in requests:
            index = int(req.attrib["index"])
            if index >= expected_max:
                print(f"Junction '{tls_id}': request index {index} > max signal index {expected_max - 1}")
                any_issues = True

if not any_issues:
    print("Alle request-Indizes passen zu den TLS-Signalindizes!")
