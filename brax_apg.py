import os
import functools

import jax
import flax.linen as nn

from brax.training.acme import running_statistics
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.apg import train as apg

from src.envs import barkour

jax.config.update("jax_enable_x64", True)



def main(argv=None):
    # Setup Environment
    env = barkour.BarkourEnv()
    eval_env = barkour.BarkourEnv()
    randomization_fn = barkour.domain_randomize

    network_factory = functools.partial(
        apg_networks.make_apg_networks,
        hidden_layer_sizes=(128,) * 4,
    )

    def progress_fn(iteration, metrics):
        print(
            f'Iteration: {iteration} \t'
            f'Episode Reward: {metrics["eval/episode_reward"]:.3f} \t'
        )
        if iteration > 0:
            print(
                f'Grad Norm: {metrics["training/grad_norm"]:.3f} \t'
                f'Param Norm: {metrics["training/params_norm"]:.3f} \t'
                f'Training Wall Time: {metrics["training/walltime"]:.3f} \t'
            )
        print('\n')

    _,_, metrics = apg.train(
        environment=env,
        policy_updates=4975,
        episode_length=1000,
        num_envs=1024,
        num_evals=200,
        learning_rate=1e-4,
        seed=0,
        normalize_observations=True,
        deterministic_eval=True,
        network_factory=network_factory,
        progress_fn=progress_fn,
        eval_env=eval_env,
        randomization_fn=randomization_fn,
    )







if __name__ == '__main__':
    main()
