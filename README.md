# MAMEDDPG
MAMEDDPG is Multi-Agent Maximum Entropy Deep Deterministic Policy Gradient. It is an algorithm that combines the Soft Actor-Critic and Multi-Agent Deep Deterministic Policy Gradient together. 

Soft actor-critic is a deep reinforcement learning framework for training maximum entropy policies in continous domains. This means that in this framework, the actor aims to maximize expected reward while also maximizing entropy—that is, succeed at the task while acting as
randomly as possible.

For the input of our algorithm, we use Convolutional Neural Network to extract the key features of the map information. 

For the real application of our algorithm， we are building a team of autonomous vehicles which could cooperate with each other and compete with the other team of vehicles with the tasks of searching and fighting in a complicated map. We are using multiagent deep reinforcement learning and self play in simulated enviroment to create optimal decentralized policy for each vehicle. In addition, in order to avoid the overfitting in the self play, we replace the original policy network model with Gaussian mixture model network to make actor to maxmize the entropy. Besides, instead of using Q-learning, we used soft Q-learning which includes a value function as a new part of original Q function. The vaue function is aimed to obtain a better policy for actor. 
