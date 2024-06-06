from typing import Any

from absl import app
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from brax.io import mjcf
from brax.mjx import pipeline


def main(argv=None):
    filename = 'models/barkour/fixed_scene.xml'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    sys = mjcf.load(filepath)
    ctrl_range = sys.actuator_ctrlrange
    ctrl_midpoint = np.mean(ctrl_range, axis=1)

    friction_param = sys.dof_damping
    gainrpm_param = sys.actuator_gainprm
    biasprm_param = sys.actuator_biasprm

    with open('params_True.pkl', 'rb') as f:
        [state_history, control_history] = pickle.load(f)

    q_measured = list(map(lambda x: x.q, state_history))
    qd_measured = list(map(lambda x: x.qd, state_history))
    q_measured = np.asarray(q_measured)
    qd_measured = np.asarray(qd_measured)
    control_history = np.asarray(control_history)

    # Create a pipeline
    init_fn = jax.jit(pipeline.init)

    def loss_fn(
        system: Any,
        state: pipeline.State,
        q_train: jnp.ndarray,
        qd_train: jnp.ndarray,
        ctrl: jnp.ndarray,
    ) -> jnp.ndarray:
        state = jax.jit(pipeline.step)(system, state, ctrl)
        loss = (
            jnp.mean(jnp.square(state.q - q_train))
            + jnp.mean(jnp.square(state.qd - qd_train))
        )
        return loss, state

    minibatch_size = 10
    grad_function = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)

    num_learning_iterations = 100
    state = init_fn(sys, q_measured[0], qd_measured[0])
    for i in range(num_learning_iterations):
        freq = 100
        position_ctrl = np.array([
            ctrl_midpoint[0],
            ctrl_midpoint[1] + 0.75 * np.sin(i / freq),
            ctrl_midpoint[2],
        ])
        _ = grad_function(sys, state, q_measured[i], qd_measured[i], position_ctrl)

        # Extract the gradients:
        friction_gradient = grad.dof_damping
        gainprm_gradient = grad.actuator_gainprm
        biasprm_gradient = grad.actuator_biasprm

        # Update the parameters:
        learning_rate = 1e-3
        gradient_descent = lambda x, dx: x - learning_rate * dx

        friction_param = jax.tree.map(
            gradient_descent, friction_param, friction_gradient,
        )
        gainrpm_param = jax.tree.map(
            gradient_descent, gainrpm_param, gainprm_gradient,
        )
        biasprm_param = jax.tree.map(
            gradient_descent, biasprm_param, biasprm_gradient,
        )

        # Update the system:
        sys = sys.replace(
            dof_damping=friction_param,
            actuator_gainprm=gainrpm_param,
            actuator_biasprm=biasprm_param,
        )

        pass


if __name__ == '__main__':
    app.run(main)
