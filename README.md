# FlappyBirdGame
This commit introduces a Deep Q-Network (DQN) implementation for training an autonomous agent to play Flappy Bird.

Key components added:

- agent.py:
  Implements the reinforcement learning agent, including action selection (epsilon-greedy policy), training loop, and interaction with the environment.

- DQN_Architecture.py:
  Defines the neural network architecture used to approximate Q-values for state-action pairs.

- Experience_replay.py:
  Implements replay buffer for storing and sampling past experiences to stabilize training and improve learning efficiency.

- game_run.py:
  Handles game execution, environment interaction, and integration with the RL agent for training/testing.

- parameters.yaml:
  Contains configurable hyperparameters such as learning rate, discount factor, epsilon decay, and batch size.

- flappybirdv0.pt:
  Saved model checkpoint for the trained DQN agent.

- flappybirdv0.log:
  Training logs capturing performance metrics across episodes.

Highlights:
- Uses experience replay for stable learning
- Implements epsilon-greedy exploration strategy
- Modular and scalable design for experimentation
- Supports training and evaluation workflows

This setup enables training an agent capable of learning optimal policies for navigating obstacles in Flappy Bird.
