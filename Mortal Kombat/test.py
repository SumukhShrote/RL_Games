import retro
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import cv2
import time

# Custom wrapper to preprocess the environment for stable baselines compatibility
class MortalKombatEnv(gym.ObservationWrapper):
    def __init__(self, game):
        super(MortalKombatEnv, self).__init__(retro.make(game=game))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_obs = cv2.resize(gray_obs, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized_obs, -1)

# Wrapper to change MultiBinary action space to continuous Box
class ContinuousToBinaryActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ContinuousToBinaryActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=env.action_space.shape, dtype=np.float32)

    def action(self, action):
        binary_action = (action > 0.5).astype(int)
        return binary_action

env = DummyVecEnv([lambda: Monitor(ContinuousToBinaryActionWrapper(MortalKombatEnv('MortalKombat3-Genesis')))])

# Load the trained PPO model
model = PPO.load("C:\\ML Projects\\ReinforcementLearning\\MortalKombat\\ppo_mortal_kombat_checkpoints\\ppo_mortal_kombat_200000_steps.zip")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    time.sleep(0.01)
    env.render()  

env.close()
