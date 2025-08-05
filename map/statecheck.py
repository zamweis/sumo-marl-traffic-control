from xml.etree import ElementTree as ET

tree = ET.parse("map.net.xml")
root = tree.getroot()

for logic in root.findall("tlLogic"):
    tl_id = logic.attrib["id"]
    for i, phase in enumerate(logic.findall("phase")):
        state = phase.attrib["state"]
        if len(state) != 57:
            print(f" Phase {i} of TLS '{tl_id}' has length {len(state)}")
