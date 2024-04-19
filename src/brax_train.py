import os

from datetime import datetime
import functools

import jax

from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model

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
        ppo.train, num_timesteps=100_000_000 // 2, num_evals=100,
        reward_scaling=1, episode_length=1000, normalize_observations=True,
        action_repeat=1, unroll_length=20, num_minibatches=32,
        num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
        entropy_cost=1e-2, num_envs=8192, batch_size=256,
        network_factory=make_networks_factory,
        randomization_fn=barkour.domain_randomize, seed=0)

    def progress(num_steps, metrics):
        print(f'{num_steps}: {metrics["eval/episode_reward"]}')

    # Reset environments since internals may be overwritten by tracers from the
    # domain randomization function.
    env = barkour.BarkourEnv()
    eval_env = barkour.BarkourEnv()
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    # Save and reload params.x
    filename = 'mjx_brax_quadruped_policy'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    model.save_params(filepath, params)


if __name__ == '__main__':
    main()
