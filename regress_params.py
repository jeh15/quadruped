from absl import app
import os
import pickle
import functools

import jax
import jax.numpy as jnp
import numpy as np

import flax.struct

import brax
from brax.io import mjcf
from brax.mjx import pipeline

import matplotlib.pyplot as plt

import time


@flax.struct.dataclass
class minibatch:
    q: jax.Array
    qd: jax.Array
    ctrl: jax.Array


def main(argv=None):
    filename = 'models/barkour/fixed_scene.xml'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    sys = mjcf.load(filepath)

    sys = sys.replace(
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
    )

    friction_param = sys.dof_damping
    target_param = 0.5239

    with open('params_True.pkl', 'rb') as f:
        [state_history, control_history] = pickle.load(f)

    q_measured = list(map(lambda x: x.q, state_history))
    qd_measured = list(map(lambda x: x.qd, state_history))
    q_measured = np.asarray(q_measured)
    qd_measured = np.asarray(qd_measured)
    control_history = np.asarray(control_history)

    # Create a pipeline
    init_fn = jax.jit(pipeline.init)
    step_fn = jax.jit(pipeline.step)

    # Run initial comparison
    state = init_fn(sys, q_measured[0], qd_measured[0])
    q_history_init = []
    qd_history_init = []
    for i in range(control_history.shape[0]):
        state = step_fn(sys, state, control_history[i])
        q_history_init.append(state.q)
        qd_history_init.append(state.qd)

    # init_vmap = jax.vmap(pipeline.init, in_axes=(None, 0, 0))
    # init_fn = jax.jit(init_vmap)

    # Test different loss functions: (Consecutive rollout, Random VMAP rollout)
    # History rollout:
    def loss_function(
        system: brax.System,
        state: pipeline.State,
        batch: minibatch,
        minibatch_size: int,
    ) -> jnp.ndarray:
        def scan_fn(carry, data):
            state = carry
            control = data
            state = step_fn(system, state, control)
            return state, (state.q, state.qd)

        _, (q, qd) = jax.lax.scan(
            f=scan_fn,
            init=state,
            xs=batch.ctrl,
            length=minibatch_size,
        )

        loss = (
            jnp.mean(jnp.square(q - batch.q))
            + jnp.mean(jnp.square(qd - batch.qd))
        )

        return loss

    # # VMAPPED:
    # @functools.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0, 0))
    # def f_map(system, state, control):
    #     next_state = step_fn(system, state, control)
    #     return next_state.q, next_state.qd

    # def loss_function(
    #     system: brax.System,
    #     states: pipeline.State,
    #     batch: minibatch,
    # ) -> jnp.ndarray:
    #     q, qd = f_map(system, states, batch.ctrl)

    #     loss = (
    #         jnp.mean(jnp.square(q - batch.q))
    #         + jnp.mean(jnp.square(qd - batch.qd))
    #     )

    #     return loss

    minibatch_size = 10
    loss_fn = functools.partial(loss_function, minibatch_size=minibatch_size)
    # loss_fn = functools.partial(loss_function)
    grad_function = jax.value_and_grad(loss_fn, allow_int=True)
    grad_fn = jax.jit(grad_function)

    num_learning_iterations = 1000
    param_history = []
    loss_history = []
    start_time = time.time()
    for iteration in range(num_learning_iterations):
        q_train = jnp.roll(q_measured, minibatch_size, axis=0)[:minibatch_size]
        qd_train = jnp.roll(qd_measured, minibatch_size, axis=0)[:minibatch_size]
        ctrl = jnp.roll(control_history, minibatch_size, axis=0)[:minibatch_size]

        batch = minibatch(q_train, qd_train, ctrl)
        states = init_fn(
            sys,
            q_train[0],
            qd_train[0],
        )

        loss, grad = grad_fn(
            sys, states, batch,
        )

        # Extract the gradients:
        friction_gradient = grad.dof_damping

        # Update the parameters:
        learning_rate = 1e-2
        gradient_descent = lambda x, dx: x - learning_rate * dx

        friction_param = jax.tree.map(
            gradient_descent, friction_param, friction_gradient,
        )
        param_history.append(friction_param)
        loss_history.append(loss)

        # Update the system:
        sys = sys.replace(
            dof_damping=friction_param,
        )

    print(f'Time taken: {time.time() - start_time}')

    # Run regressed params
    state = init_fn(sys, q_measured[0], qd_measured[0])
    q_history = []
    qd_history = []
    for i in range(control_history.shape[0]):
        state = step_fn(sys, state, control_history[i])
        q_history.append(state.q)
        qd_history.append(state.qd)

    param_history = np.asarray(param_history)
    loss_history = np.asarray(loss_history)
    q_history = np.asarray(q_history)
    qd_history = np.asarray(qd_history)
    q_history_init = np.asarray(q_history_init)
    qd_history_init = np.asarray(qd_history_init)

    fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 10))
    ax[0].plot(param_history[:, 0], color='orange', label='Abduction', linewidth=3.0)
    ax[0].plot(param_history[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    ax[0].plot(param_history[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    ax[0].hlines(target_param, 0, param_history.shape[0], color='black', linestyle='--', label='Target', linewidth=3.0)
    ax[0].legend()
    ax[0].set_title('Friction Regression')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Friction Parameter')

    ax[1].plot(loss_history, linewidth=3.0)
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Loss')

    plt.savefig('regress_params.png')

    fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 10))
    fig.suptitle('Regressed Comparison')

    ax[0].plot(q_measured[:, 0], color='orange', linestyle='--', linewidth=3.0)
    ax[0].plot(q_measured[:, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    ax[0].plot(q_measured[:, 2],  color='lightcoral', linestyle='--', linewidth=3.0)
    ax[0].plot(q_history[:, 0], color='orange', label='Abduction', linewidth=3.0)
    ax[0].plot(q_history[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    ax[0].plot(q_history[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    ax[0].legend()
    ax[0].set_title('Position')
    ax[0].set_ylabel('Position')

    ax[1].plot(qd_measured[:, 0], color='orange', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_measured[:, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_measured[:, 2], color='lightcoral', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_history[:, 0], color='orange', label='Abduction', linewidth=3.0)
    ax[1].plot(qd_history[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    ax[1].plot(qd_history[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    ax[1].legend()
    ax[1].set_title('Velocity')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Velocity')

    plt.savefig('regressed.png')

    fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 10))
    fig.suptitle('Initial Comparison')

    ax[0].plot(q_measured[:, 0], color='orange', linestyle='--', linewidth=3.0)
    ax[0].plot(q_measured[:, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    ax[0].plot(q_measured[:, 2],  color='lightcoral', linestyle='--', linewidth=3.0)
    ax[0].plot(q_history_init[:, 0], color='orange', label='Abduction', linewidth=3.0)
    ax[0].plot(q_history_init[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    ax[0].plot(q_history_init[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    ax[0].legend()
    ax[0].set_title('Position')
    ax[0].set_ylabel('Position')

    ax[1].plot(qd_measured[:, 0], color='orange', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_measured[:, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_measured[:, 2], color='lightcoral', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_history_init[:, 0], color='orange', label='Abduction', linewidth=3.0)
    ax[1].plot(qd_history_init[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    ax[1].plot(qd_history_init[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    ax[1].legend()
    ax[1].set_title('Velocity')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Velocity')

    plt.savefig('initial.png')


if __name__ == '__main__':
    app.run(main)