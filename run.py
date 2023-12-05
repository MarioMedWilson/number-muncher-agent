from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN

from env import Player


def create_PV2_env():
  return Player()

env = DummyVecEnv([create_PV2_env for _ in range(5)])

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
model = PPO('MultiInputPolicy', env,  verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.001)
model.learn(total_timesteps=1_000_000, callback=callback)
