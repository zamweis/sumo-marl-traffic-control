from sumo_rl import SumoEnvironment

env = SumoEnvironment(
    net_file="karlsruhe.net.xml",
    route_file="karlsruhe.rou.xml",
    use_gui=True,
    single_agent=True,
    reward_fn="diff-waiting-time",
    delta_time=5,
    yellow_time=2,
    min_green=5
)

print("Erkannte TLS:", env.ts_ids)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
