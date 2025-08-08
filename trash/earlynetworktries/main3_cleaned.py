from sumo_rl import SumoEnvironment

env = SumoEnvironment(
    net_file="karlsruhe3_cleaned2.net.xml",
    route_file="karlsruhe3.rou.xml",
    use_gui=True,
    single_agent=True,
    reward_fn="diff-waiting-time",
    delta_time=5,
    yellow_time=2,
    min_green=5
)

obs = env.reset()

for step in range(10):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    print(f"Step {step} | Reward: {reward}")

env.close()
