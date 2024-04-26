import functools

import jax
import flax.linen as nn
import distrax

from src.envs import barkour
from src.algorithms.ppo import network_utilities as ppo_networks
from src.distribution_utilities import ParametricDistribution
from src.algorithms.ppo.loss_utilities import loss_function
from src.algorithms.ppo.train import train


def main(argv=None):
    # Initialize Functions with Params:
    randomization_fn = barkour.domain_randomize
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_layer_sizes=(128, 128, 128, 128),
        value_layer_sizes=(256, 256),
        activation=nn.tahn,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        action_distribution=ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tahn(),
        ),
    )
    loss_fn = functools.partial(
        loss_function,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    env = barkour.BarkourEnv()

    def progress_fn(num_steps, metrics):
        print(f'Number of Steps: {num_steps} \t Episode Reward: {metrics["eval/episode_reward"]}')

    policy_generator, params, metrics = train(
        environment=env,
        num_epochs=10,
        episode_length=1000,
        num_policy_steps=20,
        action_repeat=1,
        num_envs=8192,
        num_evaluation_envs=128,
        num_evaluations=1,
        deterministic_evaluation=True,
        reset_per_epoch=True,
        seed=42,
        batch_size=256,
        num_minibatches=32,
        num_ppo_iterations=4,
        normalize_observations=True,
        network_factory=make_networks_factory,
        loss_function=loss_fn,
        progress_fn=progress_fn,
        randomization_fn=randomization_fn,
    )


if __name__ == '__main__':
    main()
