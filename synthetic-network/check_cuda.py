from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

print("CUDA verfügbar:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

env = make_vec_env("CartPole-v1", n_envs=1)

model = PPO("MlpPolicy", env, verbose=1, device="cuda")
print("Modell wurde auf Device:", model.device)

obs = env.reset()
action, _states = model.predict(obs)
print("Aktion:", action)
