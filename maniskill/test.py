import gym
import mani_skill2.envs
import create_training_data

create_training_data.swap_background(create_training_data.backgrounds[-4])

env = gym.make("BananaInBowl-v0", obs_mode="rgbd")
print("Observation space", env.observation_space)
print("Action space", env.action_space)

env.seed(0)  # specify a seed for randomness
obs = env.reset()

done = False
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()  # a display is required to render
env.close()