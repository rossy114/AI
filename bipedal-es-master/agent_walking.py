from bipedal import *

agent = Agent()

# the pre-trained weights are saved into 'weights.pkl' which you can use.
# agent.load('weights.pkl')

# play one episode
agent.play(1)
# train for 100 iterations
agent.train(2)