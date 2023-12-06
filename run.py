from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import os

from env import Player


def create_PV2_env():
  return Player()

env = DummyVecEnv([create_PV2_env for _ in range(5)])

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
model = PPO('MultiInputPolicy', env,  verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.001)
model.learn(total_timesteps=1_000_000, callback=callback)
model.save("ModelV2--PPO")
