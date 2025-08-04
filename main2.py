from sumo_rl import SumoEnvironment
import traci

print("🚦 Initialisiere SUMO-Umgebung ...")

try:
    env = SumoEnvironment(
        net_file="karlsruhe2.net.xml",
        route_file="karlsruhe2.rou.xml",
        use_gui=True,
        single_agent=True,
        reward_fn="diff-waiting-time",
        delta_time=5,
        yellow_time=2,
        min_green=5
    )
except Exception as e:
    print("❌ Fehler beim Initialisieren der Umgebung:")
    raise e

print("\n✅ SUMO-Umgebung erfolgreich erstellt.")
print(f"🔍 Anzahl erkannter TLS: {len(env.ts_ids)}")
print(f"🆔 TLS-IDs: {env.ts_ids}")

# Detaillierte Infos zu jeder TLS
for tls_id in env.ts_ids:
    try:
        logics = env.sumo.trafficlight.getAllProgramLogics(tls_id)
        print(f"\n🛑 TLS-ID: {tls_id}")
        for logic in logics:
            print(f"  ➤ Typ: {logic.type}, Phasen: {len(logic.phases)}")
            for i, phase in enumerate(logic.phases):
                print(f"    - Phase {i}: {phase.duration}s | State: {phase.state}")
            links = env.sumo.trafficlight.getControlledLinks(tls_id)
            print(f"  ➤ Gesteuerte Verbindungen (controlled links): {len(links)}")
            if len(logic.phases) < 2:
                print("  ⚠️  Warnung: Weniger als 2 Phasen – SUMO-RL könnte abstürzen.")
    except Exception as e:
        print(f"  ❌ Fehler beim Abfragen von TLS {tls_id}: {e}")

print("\n🚦 Starte Episode ...")
try:
    obs = env.reset()
except Exception as e:
    print("❌ Fehler beim Reset der Umgebung:")
    raise e

done = False
step_count = 0

while not done:
    try:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        print(f"Step {step_count:>3} | Aktion: {action} | Reward: {reward:.3f} | Done: {done}")
        step_count += 1
    except Exception as e:
        print(f"❌ Fehler bei Schritt {step_count}: {e}")
        break

print("\n✅ Simulation abgeschlossen.")
