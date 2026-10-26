import numpy as np
import gymnasium as gym


def get_dimension_from_space(space):

    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    if isinstance(space, gym.spaces.Discrete):
        return space.n


class GymnasiumEnv():

    def __init__( \
        self, env_id = "CartPole-v1",
        render_mode = "rgb_array", max_episode_steps = 1000,
        sutton_barto_reward = True) -> None:

        self.env = gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            sutton_barto_reward=sutton_barto_reward,
        )
        self.d_observation = get_dimension_from_space(self.env.observation_space)
        self.d_action = get_dimension_from_space(self.env.env.action_space)
        self.terminated = False
        self.truncated = False


    @property
    def ended(self):

        return self.terminated | self.truncated


    def step(self, action):

        observation, reward, self.terminated, self.truncated, _ = self.env.step(action)
        return observation, reward


    def close(self):
        self.env.close()

    def reset(self, seed):
        ret = self.env.reset(seed=seed)
        self.terminated = False
        self.truncated = False
        return ret
