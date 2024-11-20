import retro
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import cv2

# Custom wrapper to preprocess the environment for stable baselines compatibility
class MortalKombatEnv(gym.ObservationWrapper):
    def __init__(self, game):
        super(MortalKombatEnv, self).__init__(retro.make(game=game))
        # Convert observations to grayscale and resize to a smaller shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        # Convert to grayscale, resize, and add channel dimension
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_obs = cv2.resize(gray_obs, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized_obs, -1)

    def reset(self, **kwargs):
        # Filter out unsupported keyword arguments 
        if 'seed' in kwargs:
            kwargs.pop('seed')
        return super().reset(**kwargs)


# Wrapper to change MultiBinary action space to continuous Box
class ContinuousToBinaryActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ContinuousToBinaryActionWrapper, self).__init__(env)
        # Define continuous action space (0 to 1) with shape same as MultiBinary space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=env.action_space.shape, dtype=np.float32)
    
    def action(self, action):
        # Convert continuous actions to binary by thresholding at 0.5
        binary_action = (action > 0.5).astype(int)
        return binary_action


env = DummyVecEnv([lambda: Monitor(ContinuousToBinaryActionWrapper(MortalKombatEnv('MortalKombat3-Genesis')))])

# Create PPO model with CnnPolicy for image-based observations
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_mortal_kombat_tensorboard/")

# Set up a checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./ppo_mortal_kombat_checkpoints/', name_prefix='ppo_mortal_kombat')

# Train the model
total_timesteps = 500000  # Adjust as needed
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Save the trained model
model.save("ppo_mortal_kombat")

# Close the environment
env.close()
