from absl import app, flags
import os
import functools
import yaml
from pathlib import Path

import jax
import flax.linen as nn
import distrax
import optax

import wandb
import orbax.checkpoint as ocp

from src.envs import foraging
from src.algorithms.ppo import network_utilities as ppo_networks
from src.algorithms.ppo.loss_utilities import loss_function
from src.distribution_utilities import ParametricDistribution
from src.algorithms.ppo.train import train
from src.algorithms.ppo import checkpoint_utilities

jax.config.update("jax_enable_x64", True)
wandb.require("core")

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'orbax_multihost',
    '--experimental_orbax_use_distributed_process_id=True',
    'Enable multihost mode for orbax',
)


def train_main(argv=None):
    run = wandb.init(
        project='imsi',
        group='architecture_sweep',
    )

    # Environment Metadata:
    env_config = foraging.ForagingConfig(
        survival_reward=0.0,
        reward_scale=0.001,
        energy_cap=10.0,
        metabolic_rate=-0.1,
        work_scale=0.0,
        kinetic_energy_scale=0.0,
        foraging_scale=1.0,
        energy_capped=False,
        static_location=False,
        food_patch=False,
        foraging_rate=1.0,
        food_patch_x=2.0,
        food_patch_y=2.0,
        food_patch_r=1.0,
    )

    # Metadata:
    network_metadata = checkpoint_utilities.network_metadata(
        policy_layer_size=128,
        value_layer_size=256,
        policy_depth=3,
        value_depth=4,
        activation='nn.swish',
        kernel_init='jax.nn.initializers.orthogonal(0.01)',
        action_distribution='ParametricDistribution(distribution=distrax.Normal, bijector=distrax.Tanh())',
    )
    loss_metadata = checkpoint_utilities.loss_metadata(
        clip_coef=0.3,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=0.97,
        gae_lambda=0.95,
        normalize_advantages=True,
    )
    training_metadata = checkpoint_utilities.training_metadata(
        num_epochs=20,
        num_training_steps=20,
        episode_length=1000,
        num_policy_steps=25,
        action_repeat=1,
        num_envs=8192,
        num_evaluation_envs=128,
        num_evaluations=1,
        deterministic_evaluation=True,
        reset_per_epoch=False,
        seed=0,
        batch_size=256,
        num_minibatches=32,
        num_ppo_iterations=4,
        normalize_observations=True,
        optimizer='optax.adam(3e-4)',
    )

    # Update run config:
    run.config['enviroment_metadata'] = env_config
    run.config['network_metadata'] = network_metadata
    run.config['loss_metadata'] = loss_metadata
    run.config['training_metadata'] = training_metadata

    # Start Wandb and save metadata:
    run = wandb.init(
        project='imsi',
        group='double_integrator',
        config={
            'enviroment_metadata': env_config,
            'network_metadata': network_metadata,
            'loss_metadata': loss_metadata,
            'training_metadata': training_metadata,
        },
    )

    # Initialize Functions with Params:
    randomization_fn = None
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_layer_sizes=(network_metadata.policy_layer_size, ) * network_metadata.policy_depth,
        value_layer_sizes=(network_metadata.value_layer_size, ) * network_metadata.value_depth,
        activation=nn.swish,
        kernel_init=jax.nn.initializers.orthogonal(0.01),
        action_distribution=ParametricDistribution(
            distribution=distrax.Normal,
            bijector=distrax.Tanh(),
        ),
    )
    loss_fn = functools.partial(
        loss_function,
        clip_coef=loss_metadata.clip_coef,
        value_coef=loss_metadata.value_coef,
        entropy_coef=loss_metadata.entropy_coef,
        gamma=loss_metadata.gamma,
        gae_lambda=loss_metadata.gae_lambda,
        normalize_advantages=loss_metadata.normalize_advantages,
    )
    env = foraging.Foraging(config=env_config)
    eval_env = foraging.Foraging(config=env_config)

    def progress_fn(iteration, num_steps, metrics):
        print(
            f'Iteration: {iteration} \t'
            f'Num Steps: {num_steps} \t'
            f'Episode Reward: {metrics["eval/episode_reward"]:.3f} \t'
        )
        if num_steps > 0:
            print(
                f'Training Loss: {metrics["training/loss"]:.3f} \t'
                f'Policy Loss: {metrics["training/policy_loss"]:.3f} \t'
                f'Value Loss: {metrics["training/value_loss"]:.3f} \t'
                f'Entropy Loss: {metrics["training/entropy_loss"]:.3f} \t'
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
        episode_length=training_metadata.episode_length,
        num_policy_steps=training_metadata.num_policy_steps,
        action_repeat=training_metadata.action_repeat,
        num_envs=training_metadata.num_envs,
        num_evaluation_envs=training_metadata.num_evaluation_envs,
        num_evaluations=training_metadata.num_evaluations,
        deterministic_evaluation=training_metadata.deterministic_evaluation,
        reset_per_epoch=training_metadata.reset_per_epoch,
        seed=training_metadata.seed,
        batch_size=training_metadata.batch_size,
        num_minibatches=training_metadata.num_minibatches,
        num_ppo_iterations=training_metadata.num_ppo_iterations,
        normalize_observations=training_metadata.normalize_observations,
        network_factory=make_networks_factory,
        optimizer=eval(training_metadata.optimizer),
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

    run.finish()


def main(argv=None):
    config = yaml.safe_load(Path('config.yaml').read_text())
    sweep_id = wandb.sweep(sweep=config, project='imsi')
    wandb.agent(sweep_id, function=train_main)


if __name__ == '__main__':
    app.run(main)