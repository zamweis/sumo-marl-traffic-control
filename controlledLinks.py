import traci
import sumolib
from sumo_rl import SumoEnvironment

sumoBinary = sumolib.checkBinary("sumo")
traci.start([sumoBinary, "-n", "karlsruhe.net.xml"])
tls_id = "cluster_10568487727_10568487728_10568487731_14795133_#10more"
controlled_links = traci.trafficlight.getControlledLinks(tls_id)
print(f"🔧 TLS '{tls_id}' controls {len(controlled_links)} signal indices.")
traci.close()
