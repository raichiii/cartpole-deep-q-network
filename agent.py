from typing import Type
from collections import deque
import numpy as np
import torch
from torch import nn

from gymnasium_env import GymnasiumEnv


class Agent():

    def __init__( \
        self, env: GymnasiumEnv,
        Q_function: nn.Module, loss_fn,
        optimizer, lr_scheduler,
        lr = 0.5, gamma = 0.99,
        step_size_batch = 1000, rng = np.random.default_rng()) -> None:

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.eye_d_action = np.eye(env.d_action)
        self.rng = rng

        self.Q_function = Q_function
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train()

        self.step_size_batch = step_size_batch
        self.arange_steps = np.arange(step_size_batch)
        self.observations = deque()
        self.actions = deque()
        self.rewards = deque()
        self.observations_next = deque()
        self.losses = []

    def add_step_buffer(self, step):

        observation, action, reward, observation_next = step

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations_next.append(observation_next)
        if self.step_size_batch < len(self.observations):
            self.observations.popleft()
            self.actions.popleft()
            self.rewards.popleft()
            self.observations_next.popleft()

    def update_Q_function(self):
        """Update Q function"""

        if len(self.observations) < self.step_size_batch:
            return

        observations = np.array(self.observations)
        rewards = np.array(self.rewards)
        observations_next = np.array(self.observations_next)

        X_next = torch.tensor(observations_next, dtype=torch.float32).to("cuda")
        self.Q_function.eval()
        Q_next = self.Q_function(X_next).cpu().detach().numpy()
        Max_Q_next = Q_next.max(axis=1) # Get max action value on the next state

        X = torch.tensor(observations, dtype=torch.float32).to("cuda")
        Q = rewards + self.gamma * Max_Q_next
        Y = torch.tensor(Q).to("cuda")
        self.Q_function.train()
        pred = self.Q_function(X)[self.arange_steps, self.actions]
        loss = self.loss_fn(pred, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.losses.append(loss.item())

    def __get_epsilon_greedy_action(self, observation, episode, *args, **kwargs):
        """Decide action with ε-greedy method
        """

        epsilon = 0.5 * (1 / (episode + 1)) # Epsilon decreases as the number of episodes increases

        if epsilon <= self.rng.uniform(0, 1):
            return self.__get_optimal_action(observation=observation)
        else:
            return self.rng.choice(self.env.d_action)

    def __get_optimal_action(self, observation, *args, **kwargs):
        """Get optimal action
        """

        X = torch.tensor(observation, dtype=torch.float32).to("cuda")
        self.Q_function.eval()
        Qs = self.Q_function(X).cpu().detach().numpy()
        action = np.argmax(Qs)
        return action

    def train(self):
        self.Q_function.train()
        self.get_action = self.__get_epsilon_greedy_action

    def eval(self):
        self.Q_function.eval()
        self.get_action = self.__get_optimal_action
