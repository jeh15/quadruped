from absl import app, flags
import os

import jax
import jax.numpy as jnp

from brax.io import html

from src.envs import foraging
from src.algorithms.ppo.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)


def main(argv=None):
    # Load from Env:
    env_config = foraging.ForagingConfig(
        survival_reward=0.0,
        reward_scale=0.001,
        energy_cap=10.0,
        metabolic_rate=-0.02,
        work_scale=0.0,
        kinetic_energy_scale=0.0,
        foraging_scale=1.0,
        energy_capped=True,
        static_location=True,
        food_patch=False,
        foraging_rate=1.0,
        food_patch_x=2.0,
        food_patch_y=2.0,
        food_patch_r=1.0,
    )
    env = foraging.Foraging(
        config=env_config, filename='double_integrator/foraging_scene.xml',
    )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Load Policy:
    make_policy, params = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params)
    inference_fn = jax.jit(inference_function)

    # Initialize Simulation:
    key = jax.random.key(0)
    state = reset_fn(key)

    num_steps = 1000
    states = []
    print(f"Food Patch: {state.info['food_patch']}")
    print(f"Initial Position: {state.pipeline_state.q}")
    for i in range(num_steps):
        action_rng, key = jax.random.split(key)
        action, _ = inference_fn(state.obs, action_rng)
        state = step_fn(state, action)
        states.append(state.pipeline_state)
        if i % 50 == 0:
            print(f"Reward: {state.reward}")
            print(f"Observation: {state.obs}")

    states = list(map(lambda x: x.replace(contact=None), states))

    # Visualize:
    html_string = html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
        states=states,
        height='100vh',
        colab=False,
    )
    html_path = os.path.join(
        os.path.dirname(__file__),
        "visualization/foraging.html",
    )

    with open(html_path, 'w') as f:
        f.write(html_string)


if __name__ == '__main__':
    app.run(main)