from logging import getLogger
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import matplotlib.pyplot as plt


from gymnasium_env import GymnasiumEnv
from agent import Agent
from episode_controller import EpisodeController
from util import seed_everything, make_gif_from_ndarray, plot_simply


logger = getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Parameters
    env_id = cfg.env_id
    lr = cfg.lr
    gamma = cfg.gamma
    sutton_barto_reward = cfg.sutton_barto_reward
    max_episode_size = cfg.max_episode_size
    max_episode_steps = cfg.max_episode_steps
    step_size_batch = cfg.step_size_batch
    step_size_thres_succeeded = cfg.step_size_thres_succeeded
    n_max_continuous_success = cfg.n_max_continuous_success
    seed = cfg.seed
    fname_gif = cfg.display.fname_gif
    duration_gif = cfg.display.duration_gif
    model = cfg.model
    loss_fn = cfg.loss_fn
    optimizer = cfg.optimizer
    lr_scheduler = cfg.lr_scheduler

    # Confirm conditions
    assert step_size_thres_succeeded < max_episode_steps


    rng = seed_everything(seed)

    # Get instances
    env = GymnasiumEnv(env_id=env_id, max_episode_steps=max_episode_steps, sutton_barto_reward=sutton_barto_reward)
    Q_function = instantiate(model, d_input=env.d_observation, d_output=env.d_action).to("cuda")
    loss_fn = instantiate(loss_fn)
    optimizer = instantiate(optimizer, params=Q_function.parameters())
    lr_scheduler = instantiate(lr_scheduler, optimizer=optimizer)
    agent = Agent(
        env=env, Q_function=Q_function, loss_fn=loss_fn,
        optimizer=optimizer, lr_scheduler=lr_scheduler, lr=lr,
        gamma=gamma, step_size_batch=step_size_batch,
        rng=rng,
    )
    episode_ctrl = EpisodeController(env=env, agent=agent)

    episode_ctrl.reset()
    n_continuous_success = 0
    step_size_episodes = []

    for i_episode in tqdm(range(max_episode_size)):

        # Run a episode
        step_size = episode_ctrl.run(record=False)
        step_size_episodes.append(step_size)

        if (i_episode + 1) % 20 == 0:
            plot_simply(
                x=step_size_episodes,
                fname="StepSize_CartPole_DQN",
                yscale="log",
                xlabel="Episode",
                ylabel="Step size succeeded",
            )
            plot_simply(
                x=agent.losses,
                fname="Loss_CartPole_DQN",
                yscale="log",
                xlabel="Total step size",
                ylabel="Loss",
            )

        if step_size_thres_succeeded <= step_size:
            n_continuous_success += 1

        else:
            n_continuous_success = 0

        if n_continuous_success == n_max_continuous_success:
            break

    plot_simply(
        x=step_size_episodes,
        fname="StepSize_CartPole_DQN",
        yscale="log",
        xlabel="Episode",
        ylabel="Step size succeeded",
    )
    plot_simply(
        x=agent.losses,
        fname="Loss_CartPole_DQN",
        yscale="log",
        xlabel="Total step size",
        ylabel="Loss",
    )
    episode_ctrl.reset()
    agent.eval()
    episode_ctrl.run(record=True)
    make_gif_from_ndarray(
        episode_ctrl.frames,
        fname_gif,
        duration=duration_gif,
    )

    env.close()


if __name__ == "__main__":
    main()
