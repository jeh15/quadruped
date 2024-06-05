from absl import app
import functools
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
    initial_q = jnp.array([
        0.0, 0.5, 1.0,
    ])
    initial_qd = jnp.zeros_like(initial_q)
    state = jax.jit(pipeline.init)(sys, initial_q, initial_qd)
    step_fn = jax.jit(pipeline.step)

    states = []

    def loss_fn(sys, state, ctrl, q_measured, qd_measured):
        q = []
        qd = []
        for i in range(unroll_length):
            state = step_fn(sys, state, ctrl[i])
            q.append(state.q)
            qd.append(state.qd)
        q = jnp.asarray(q)
        qd = jnp.asarray(qd)
        loss = (
            jnp.mean(jnp.square(q - q_measured))
            + jnp.mean(jnp.square(qd - qd_measured))
        )
        return loss, state

    grad_function = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)
    grad_fn = jax.jit(grad_function)

    num_learning_iterations = 250
    for iteration in range(num_learning_iterations):
        unroll_length = 25
        q_train = np.roll(q_measured, unroll_length)[:unroll_length]
        qd_train = np.roll(qd_measured, unroll_length)[:unroll_length]
        ctrl = np.roll(control_history, unroll_length)[:unroll_length]
        state = jax.jit(pipeline.init)(sys, q_train[0], qd_train[0])
        ((loss, state), grad) = grad_fn(
            sys, state, ctrl, q_train, qd_train,
        )

        # Extract the gradients:
        friction_gradient = grad.dof_damping
        gainrpm_gradient = grad.actuator_gainprm
        biasprm_gradient = grad.actuator_biasprm

        # Update the parameters:
        learning_rate = 1e-3
        gradient_descent = lambda x, dx: x - learning_rate * dx

        friction_param = jax.tree.map(
            gradient_descent, friction_param, friction_gradient,
        )
        gainrpm_param = jax.tree.map(
            gradient_descent, gainrpm_param, gainrpm_gradient,
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

        states.append(state)

        pass


if __name__ == '__main__':
    app.run(main)
