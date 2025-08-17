import sumolib

net = sumolib.net.readNet("map.net.xml")
edges = net.getEdges()

fringe_edges = []
for e in edges:
    # Filter: keine eingehenden Verbindungen (Start am Rand)
    if len(e.getIncoming()) == 0:
        # Optional: nur größere Straßen
        if len(e.getLanes()) >= 2:
            fringe_edges.append(e.getID())

print(fringe_edges)
