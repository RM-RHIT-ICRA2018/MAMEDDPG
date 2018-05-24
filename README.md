# MAMEDDPG
MAMEDDPG is Multi-Agent Maximum Entropy Deep Deterministic Policy Gradient. It is an algorithm that combines the Soft Actor-Critic and Multi-Agent Deep Deterministic Policy Gradient (MADDPG) together. 

Soft actor-critic is a deep reinforcement learning framework for training maximum entropy policies in continous domains. This means that in this framework, the actor aims to maximize expected reward while also maximizing entropy—that is, succeed at the task while acting as
randomly as possible.

For the MADDPG part:
We modified the current MADDPG algorithm to train our policy, the following is the brief introduction to the MADDPG algorithm in the paper Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments:
In our settings, we have three agents, let’s name them op1, op2 and ag, op1 and op2 are the two opponents robots while ag is our own robot. We have simplified the action all robot can do into the following two categories: (1) changing direction of the shooting area of gun turret to the left 45 degree, to the right 45 degree or stay the same direction. (2) applying an acceleration in to one of the eight
directions. Thus, we have in total 24 discrete action choice. It should be noted that it is totally possible to extend the system to a continuous action space, which has been implemented in MADDPG. (This is because the backbone of MADDPG is DDPG,
which use a policy to sample from continuous action space). But we have decided to use discrete action space in the hope that we could accelerate the training process by decreasing the space of Q function.

Also, all of the agent have the same observation space: the self absolute position, self speed, the relative position and relative speed of other two robots. Again, this is based on the assumption that our robot could capture the position of the all opponent robots, which will be further explained in the section on enemy detection.
For both the policy of each agent and the Q function, we used MLP neural network as approximation. The policy network detailed structure is as below:





The Q network is as below:





Each agent will held its own policy network and Q value network. This is because the reward function for each agent will be different, which will cause the Q value be different.

For the input of our algorithm, we use Convolutional Neural Network to extract the key features of the map information. 

For the real application of our algorithm， we are building a team of autonomous vehicles which could cooperate with each other and compete with the other team of vehicles with the tasks of searching and fighting in a complicated map. We are using multiagent deep reinforcement learning and self play in simulated enviroment to create optimal decentralized policy for each vehicle. In addition, in order to avoid the overfitting in the self play, we replace the original policy network model with Gaussian mixture model network to make actor to maxmize the entropy. Besides, instead of using Q-learning, we used soft Q-learning which includes a value function as a new part of original Q function. The vaue function is aimed to obtain a better policy for actor. 
