from datetime import datetime
import functools

import jax

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

from matplotlib import pyplot as plt

import barkour

jax.config.update("jax_enable_x64", True)


def main(argv=None):
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 10)

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128),
    )
    train_fn = functools.partial(
        ppo.train, num_timesteps=100_000_000, num_evals=100,
        reward_scaling=1, episode_length=1000, normalize_observations=True,
        action_repeat=1, unroll_length=20, num_minibatches=32,
        num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
        entropy_cost=1e-2, num_envs=8192, batch_size=256,
        network_factory=make_networks_factory,
        randomization_fn=barkour.domain_randomize, seed=0)

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    max_y, min_y = 40, 0

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])

        plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={y_data[-1]:.3f}')

        plt.errorbar(
            x_data, y_data, yerr=ydataerr, color='blue',
        )
        plt.show(block=False)
        plt.pause(0.001)

    # Reset environments since internals may be overwritten by tracers from the
    # domain randomization function.
    env = barkour.BarkourEnv()
    eval_env = barkour.BarkourEnv()
    make_inference_fn, params, _= train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')


if __name__ == '__main__':
    main()
