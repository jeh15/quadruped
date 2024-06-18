from absl import app
import os
import functools

import jax
import flax.linen as nn
import distrax
import optax

import wandb
import orbax.checkpoint as ocp

from src.envs import barkour
from src.algorithms.apg import network_utilities as apg_networks
from src.algorithms.apg.loss_utilities import loss_function
from src.distribution_utilities import ParametricDistribution
from src.algorithms.apg.train import train
from src.algorithms.apg import checkpoint_utilities

jax.config.update("jax_enable_x64", True)


def main(argv=None):
    # Metadata:
    network_metadata = checkpoint_utilities.network_metadata(
        policy_layer_size=128,
        policy_depth=4,
        activation='nn.elu',
        kernel_init='jax.nn.initializers.orthogonal(0.01)',
        action_distribution='ParametricDistribution(distribution=distrax.Normal, bijector=distrax.Tanh())',
    )
    loss_metadata = checkpoint_utilities.loss_metadata(
        horizon_length=32,
    )

    training_metadata = checkpoint_utilities.training_metadata(
        num_epochs=20,
        num_training_steps=25,
        horizon_length=loss_metadata.horizon_length,
        episode_length=1000,
        action_repeat=1,
        num_envs=1024,
        num_evaluation_envs=128,
        num_evaluations=1,
        deterministic_evaluation=True,
        reset_per_epoch=False,
        seed=0,
        normalize_observations=True,
        optimizer='optax.chain(optax.clip_by_global_norm(1e9), optax.clip(1.0),optax.adam(learning_rate=learning_rate,b1=0.7,b2=0.95))',
        learning_rate='optax.exponential_decay(init_value=1e-4,transition_steps=1,decay_rate=0.997)',
        use_float64=False,
    )

    # Start Wandb and save metadata:
    run = wandb.init(
        config={
            'network_metadata': network_metadata,
            'loss_metadata': loss_metadata,
            'training_metadata': training_metadata,
        },
    )

    # Initialize Functions with Params:
    randomization_fn = barkour.domain_randomize
    make_networks_factory = functools.partial(
        apg_networks.make_apg_networks,
        policy_layer_sizes=(network_metadata.policy_layer_size, ) * network_metadata.policy_depth,
        activation=nn.elu,
        kernel_init=jax.nn.initializers.orthogonal(0.01),
        action_distribution=ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tanh(),
        ),
    )
    loss_fn = functools.partial(
        loss_function,
        horizon_length=loss_metadata.horizon_length,
    )

    # Create Optimizer: (Include Gradient Clipping)
    learning_rate = optax.exponential_decay(
        init_value=1e-4,
        transition_steps=1,
        decay_rate=0.997
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1e9),
        optax.clip(1.0),
        optax.adam(
            learning_rate=learning_rate,
            b1=0.7,
            b2=0.95,
        ),
    )

    env = barkour.BarkourEnv()
    eval_env = barkour.BarkourEnv()

    def progress_fn(iteration, num_steps, metrics):
        print(
            f'Iteration: {iteration} \t'
            f'Num Steps: {num_steps} \t'
            f'Episode Reward: {metrics["eval/episode_reward"]:.3f} \t'
        )
        if num_steps > 0:
            print(
                f'Grad Norm: {metrics["training/grad_norm"]:.3f} \t'
                f'Param Norm: {metrics["training/params_norm"]:.3f} \t'
                f'Training Wall Time: {metrics["training/walltime"]:.3f} \t'
            )
        print('\n')

    # Setup Checkpoint Manager:
    manager_options = checkpoint_utilities.default_checkpoint_options()
    checkpoint_direrctory = os.path.join(
        os.path.dirname(__file__),
        f"checkpoints/{run.name}",
    )
    manager = ocp.CheckpointManager(
        directory=checkpoint_direrctory,
        options=manager_options,
        item_names=(
            'train_state',
            'network_metadata',
            'loss_metadata',
            'training_metadata',
        ),
    )
    checkpoint_fn = functools.partial(
        checkpoint_utilities.save_checkpoint,
        manager=manager,
        network_metadata=network_metadata,
        loss_metadata=loss_metadata,
        training_metadata=training_metadata,
    )

    train_fn = functools.partial(
        train,
        num_epochs=training_metadata.num_epochs,
        num_training_steps=training_metadata.num_training_steps,
        horizon_length=loss_metadata.horizon_length,
        episode_length=training_metadata.episode_length,
        action_repeat=training_metadata.action_repeat,
        num_envs=training_metadata.num_envs,
        num_evaluation_envs=training_metadata.num_evaluation_envs,
        num_evaluations=training_metadata.num_evaluations,
        deterministic_evaluation=training_metadata.deterministic_evaluation,
        reset_per_epoch=training_metadata.reset_per_epoch,
        seed=training_metadata.seed,
        normalize_observations=training_metadata.normalize_observations,
        network_factory=make_networks_factory,
        optimizer=optimizer,
        loss_function=loss_fn,
        progress_fn=progress_fn,
        randomization_fn=randomization_fn,
        checkpoint_fn=checkpoint_fn,
        wandb=run,
    )

    policy_generator, params, metrics = train_fn(
        environment=env,
        evaluation_environment=eval_env,
    )

    wandb.finish()


if __name__ == '__main__':
    app.run(main)
