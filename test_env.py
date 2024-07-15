from absl import app, flags
import os

import jax
import jax.numpy as jnp

from brax.io import html

from src.envs import foraging
from src.algorithms.ppo.load_utilities import load_policy

jax.config.update("jax_enable_x64", True)


def main(argv=None):
    # Load from Env:
    env = foraging.Foraging(filename='double_integrator/foraging_scene.xml')

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # Initialize Simulation:
    key = jax.random.key(0)
    state = reset_fn(key)

    num_steps = 1000
    states = []
    print(f"Food Patch: {state.info['food_patch']}")
    print(f"Initial Position: {state.pipeline_state.q}")
    for i in range(num_steps):
        action = jnp.array([0.5, -0.5])
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
        "visualization/test.html",
    )

    with open(html_path, 'w') as f:
        f.write(html_string)


if __name__ == '__main__':
    app.run(main)
