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
    env = foraging.Foraging(filename='double_integrator/foraging_scene.xml')

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
    for _ in range(num_steps):
        action_rng, key = jax.random.split(key)
        action, _ = inference_fn(state.obs, action_rng)
        state = step_fn(state, action)
        states.append(state.pipeline_state)
        print("Reward:", state.reward)

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
