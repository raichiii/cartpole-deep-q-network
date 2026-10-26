from logging import getLogger
import numpy as np
from gymnasium.wrappers import RecordVideo


logger = getLogger(__name__)


class EpisodeController:
    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent
        self.frames = []
        self.n_step_episodes = []
        self.i_episode = 0

    def run(self, record = False):
        if record:
            self.env.env = RecordVideo(self.env.env, video_folder="video")

        self.frames = []
        state, _ = self.env.reset(seed=self.i_episode)

        i_step = 0
        while not self.env.ended:

            if record:
                # Stock frames for gif
                self.frames.append(self.env.env.render())

            # Get action
            action = self.agent.get_action(state, self.i_episode)

            state_next, reward = self.env.step(action)
            self.agent.add_step_buffer([state, action, reward, state_next])
            self.agent.update_Q_function()

            state = state_next
            i_step += 1

        self.n_step_episodes.append(i_step)
        self.i_episode += 1

        return i_step

    def reset(self):
        self.frames = []
