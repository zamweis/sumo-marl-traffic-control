from xml.etree import ElementTree as ET

# Manuell gepflegte Dictionary mit {TLS-ID: Anzahl controlledLinks}
controlled_links = {
    "1720933516": 6,
    "3538953167": 2,
    "3664415977": 10,
    "cluster_14795187_1720919996_2670370290_2670370291": 11,
    "cluster_14795804_55474925_6655074904_765746891_#1more": 49,
    "cluster_15431428_1719671850_1720917935": 20,
    "cluster_1590912233_3664415976_5083348337_5083348350": 11,
    "cluster_1692973685_1692973722_1718084055_1718084058_#11more": 36,
    "cluster_1729190097_3687504105": 8,
    "cluster_1744031943_5131521735": 10,
    "joinedS_1623835169_cluster_1137679587_1626739216_1728272870_1728272909_#17more": 33,
    "joinedS_309108716_cluster_11001804363_1125509937_12515596172_1784859792_#5more": 14,
    "joinedS_5092985445_cluster_1590912226_2911376263": 10,
    # ggf. mehr hinzufügen
}

tree = ET.parse("karlsruhe.net.xml")
root = tree.getroot()
changed = False

for logic in root.findall("tlLogic"):
    tl_id = logic.attrib["id"]
    if tl_id not in controlled_links:
        continue

    correct_len = controlled_links[tl_id]
    for phase in logic.findall("phase"):
        state = phase.attrib["state"]
        if len(state) != correct_len:
            new_state = state[:correct_len].ljust(correct_len, 'r')
            print(f" Fixing {tl_id}: {len(state)} → {correct_len}")
            phase.attrib["state"] = new_state
            changed = True

if changed:
    tree.write("karlsruhe_fixed.net.xml")
    print(" Bereinigte Datei gespeichert: karlsruhe_fixed.net.xml")
else:
    print(" Alle Phasen bereits korrekt.")
