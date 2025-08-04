from sumo_rl import SumoEnvironment
import traci
import os

def test_tls(tls_id):
    try:
        env = SumoEnvironment(
            net_file="karlsruhe.net.xml",
            route_file="karlsruhe.rou.xml",
            use_gui=False,
            single_agent=True
        )
        env.ts_ids = [tls_id]
        env.reset()
        env.close()
        return True
    except Exception as e:
        print(f" TLS {tls_id} nicht gültig: {e}")
        return False

# Alle TLS holen
try:
    env = SumoEnvironment(
        net_file="karlsruhe.net.xml",
        route_file="karlsruhe.rou.xml",
        use_gui=False,
        single_agent=True
    )
    all_tls = env.ts_ids
    env.close()
except Exception as e:
    print(" Konnte TLS nicht auslesen:", e)
    all_tls = []

print(f" Teste {len(all_tls)} TLS auf Gültigkeit...\n")
valid_tls = []

for tls_id in all_tls:
    if test_tls(tls_id):
        valid_tls.append(tls_id)

print("\n Gültige TLS:")
print(valid_tls)
