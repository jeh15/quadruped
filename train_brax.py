from absl import app, flags, logging
import os
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import optax

import wandb
import orbax.checkpoint as ocp

from brax.io import html

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from src.envs import unitree_go2_height_control as unitree_go2

# from src.algorithms.ppo import network_utilities as ppo_networks
# from src.algorithms.ppo.loss_utilities import loss_function
# from src.distribution_utilities import ParametricDistribution
# from src.algorithms.ppo.train import train
# from src.algorithms.ppo import checkpoint_utilities
# from src.algorithms.ppo.load_utilities import load_checkpoint

# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_enable_triton_softmax_fusion=true '
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_async_collectives=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
#     '--xla_gpu_enable_highest_priority_async_stream=true '
# )

jax.config.update("jax_enable_x64", True)

logging.set_verbosity(logging.FATAL)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)
flags.DEFINE_integer(
    'checkpoint_iteration', None, 'Desired checkpoint iteration.', short_name='i',
)
flags.DEFINE_string(
    'tag', '', 'Tag for wandb run.', short_name='t',
)


"""
    TODO(jeh15): Remove Control Range on Actuators. Change Knee Force Range and add transmission.
"""

def main(argv=None):

    # Config:
    reward_config = unitree_go2.RewardConfig(
        # Rewards:
        tracking_height=2.0,
        tracking_height_error=-5.0,
        # Orientation Regularization Terms:
        angular_xy_velocity=-0.0,
        orientation_regularization=-0.0,
        pose_regularization=0.0,
        # Energy Regularization Terms:
        torque=-0.0,
        action_rate=-0.01,
        mechanical_power=-0.0,
        acceleration=-0.0,
        # Auxilary Terms:
        termination=-1.0,
        # Gait Terms:
        foot_slip=-0.0,
        # Hyperparameter for exponential kernel:
        kernel_sigma=0.1,
    )

    env = unitree_go2.UnitreeGo2Env(config=reward_config)

    network_fn = ppo_networks.make_ppo_networks
    network_factory = functools.partial(
        network_fn,
        policy_hidden_layer_sizes=(128, 128, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

    def progress_fn(iteration, metrics):
        print(
            f'Iteration: {iteration} \t'
            f'Episode Reward: {metrics["eval/episode_reward"]:.3f} \t'
        )
        if iteration > 0:
            print(
                f'Training Wall Time: {metrics["training/walltime"]:.3f} \t'
            )
        print('\n')

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory=network_factory,
        seed=42,
        num_eval_envs=128,
    )

    policy_generator, params, metrics = train_fn(
        environment=env,
        progress_fn=progress_fn,
    )

    inference_function = policy_generator(params, deterministic=True)
    inference_fn = jax.jit(inference_function)

    env = unitree_go2.UnitreeGo2Env()
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    num_steps = 1000

    prone_states = []
    state = reset_fn(subkey)
    for i in range(num_steps):
        state.info['command'] = jnp.array([0.0])
        key, subkey = jax.random.split(key)
        action, _ = inference_fn(state.obs, subkey)
        state = step_fn(state, action)
        prone_states.append(state.pipeline_state)

    tall_states = []
    state = reset_fn(subkey)
    for i in range(num_steps):
        state.info['command'] = jnp.array([0.5])
        key, subkey = jax.random.split(key)
        action, _ = inference_fn(state.obs, subkey)
        state = step_fn(state, action)
        tall_states.append(state.pipeline_state)

    # Generate HTML:
    html_string = html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
        states=prone_states,
        height="100vh",
        colab=False,
    )

    html_path = os.path.join(
        os.path.dirname(__file__),
        "visualization/prone_visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)

    html_string = html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
        states=tall_states,
        height="100vh",
        colab=False,
    )

    html_path = os.path.join(
        os.path.dirname(__file__),
        "visualization/tall_visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
