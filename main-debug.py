import os
import logging
from sumo_rl import SumoEnvironment
import traci

logging.basicConfig(level=logging.DEBUG)

os.environ["SUMO_LOGLEVEL"] = "3"  # SUMO intern loggt mehr

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

print("✅ Umgebung erfolgreich initialisiert.")
print("🔧 Erkannte TLS:", env.ts_ids)

try:
    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        step_count += 1

        if step_count % 10 == 0:
            print(f"Step {step_count} - Reward: {reward}")
            for tls_id in env.ts_ids:
                try:
                    state = traci.trafficlight.getRedYellowGreenState(tls_id)
                    phase = traci.trafficlight.getPhase(tls_id)
                    print(f"  TLS {tls_id} Phase: {phase}, State: {state}")
                except Exception as e:
                    print(f"  ⚠️ Fehler bei TLS {tls_id}: {e}")

    print(f"🏁 Simulation beendet nach {step_count} Schritten.")

except Exception as e:
    print("❌ Fehler während der Simulation:")
    raise e
