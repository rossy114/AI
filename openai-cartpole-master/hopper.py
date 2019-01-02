import gym
env = gym.make('Hopper-v2')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

from gym import envs
print(envs.registry.all())