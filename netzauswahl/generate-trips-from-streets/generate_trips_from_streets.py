import xml.etree.ElementTree as ET

trip_flows = [
    ("-1157505360", "503190615#0", 600),
    ("-1003269855#2", "86548565#1", 400),
    # weitere Paare hinzufügen...
]

routes = ET.Element("routes")
ET.SubElement(routes, "vType", {
    "id": "car", "accel": "2.6", "decel": "4.5", "length": "5.0",
    "maxSpeed": "13.9", "sigma": "0.5", "color": "1,0,0"
})

for idx, (frm, to, num) in enumerate(trip_flows, start=1):
    ET.SubElement(routes, "flow", {
        "id": f"flow{idx}", "type": "car",
        "begin": "0", "end": "3600", "number": str(num),
        "from": frm, "to": to,
        "departLane": "best", "departSpeed": "max"
    })

tree = ET.ElementTree(routes)
tree.write("flows.xml", encoding="utf-8", xml_declaration=True)
print("flows.xml erstellt")
