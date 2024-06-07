from absl import app
import os
import pickle
import functools

import jax
import jax.numpy as jnp
import numpy as np

import flax.struct
import optax

import brax
from brax.io import mjcf
from brax.mjx import pipeline

import matplotlib.pyplot as plt

import time

jax.config.update('jax_enable_x64', True)


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

    # Initialize the parameter and optimizer:
    solver = optax.adam(learning_rate=1e-3)
    params = jnp.array(friction_param)
    opt_state = solver.init(params)

    with open('train_params.pkl', 'rb') as f:
        [state_history, control_history] = pickle.load(f)

    q_measured = list(map(lambda x: x.q, state_history))
    qd_measured = list(map(lambda x: x.qd, state_history))
    q_measured = np.asarray(q_measured)
    qd_measured = np.asarray(qd_measured)
    control_history = np.swapaxes(
        np.asarray(control_history), 0, 1,
    )

    # Get the number of time steps and trials
    num_time_steps = q_measured.shape[0]
    num_trials = q_measured.shape[1]
    num_dofs = q_measured.shape[2]
    num_act = control_history.shape[-1]

    # Structure Data into Minibatches:
    minibatch_size = 10
    num_batches = num_time_steps // minibatch_size
    q_batch = np.reshape(
        q_measured, (-1, minibatch_size, num_trials, num_dofs),
    )
    qd_batch = np.reshape(
        qd_measured, (-1, minibatch_size, num_trials, num_dofs),
    )
    ctrl_batch = np.reshape(
        control_history, (-1, minibatch_size, num_trials, num_act),
    )

    # Axis order: (batch, trial, minibatch, dof)
    q_batch = np.swapaxes(q_batch, 1, 2)
    qd_batch = np.swapaxes(qd_batch, 1, 2)
    ctrl_batch = np.swapaxes(ctrl_batch, 1, 2)

    # Shuffle Data:
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    # Use same key to preserve order:
    q_train = jax.random.permutation(subkey, q_batch, axis=0)
    qd_train = jax.random.permutation(subkey, qd_batch, axis=0)
    ctrl_train = jax.random.permutation(subkey, ctrl_batch, axis=0)

    # VMAP and Jit the pipeline functions:
    vmap_init_fn = jax.vmap(pipeline.init, in_axes=(None, 0, 0))
    init_fn = jax.jit(vmap_init_fn)

    vmap_step_fn = jax.vmap(pipeline.step, in_axes=(None, 0, 0))
    step_fn = jax.jit(vmap_step_fn)

    # Run initial comparison of random trial:
    random_idx = jax.random.randint(subkey, (), minval=0, maxval=num_trials-1)
    q_initial_trial = q_measured[:, random_idx]
    qd_initial_trial = qd_measured[:, random_idx]
    ctrl_initial_trial = control_history[:, random_idx]
    state = jax.jit(pipeline.init)(sys, q_initial_trial[0], qd_initial_trial[0])
    q_history_init = []
    qd_history_init = []
    for i in range(control_history.shape[0]):
        state = jax.jit(pipeline.step)(sys, state, ctrl_initial_trial[i])
        q_history_init.append(state.q)
        qd_history_init.append(state.qd)

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

        # Reshape the data: axis -> (trials, time, dof)
        q = jnp.swapaxes(q, 0, 1)
        qd = jnp.swapaxes(qd, 0, 1)

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

    loss_fn = functools.partial(loss_function, minibatch_size=minibatch_size)
    grad_function = jax.value_and_grad(loss_fn, allow_int=True)
    grad_fn = jax.jit(grad_function)

    num_learning_iterations = 100
    param_history = []
    loss_history = []
    start_time = time.time()
    for i in range(num_learning_iterations):
        # Shuffle Data and use same key to preserve order:
        key, subkey = jax.random.split(subkey)
        q_train = jax.random.permutation(subkey, q_batch, axis=0)
        qd_train = jax.random.permutation(subkey, qd_batch, axis=0)
        ctrl_train = jax.random.permutation(subkey, ctrl_batch, axis=0)
        for j in range(num_batches):
            # Get the minibatch:
            q = q_train[j]
            qd = qd_train[j]
            ctrl = jnp.swapaxes(ctrl_train[j], 0, 1)
            batch = minibatch(
                q, qd, ctrl,
            )
            states = init_fn(
                sys,
                batch.q[:, 0],
                batch.qd[:, 0],
            )

            loss, grad = grad_fn(
                sys, states, batch,
            )

            # Extract the gradients:
            friction_gradient = grad.dof_damping

            # Update the parameters:
            updates, opt_state = solver.update(
                friction_gradient, opt_state, params,
            )
            params = optax.apply_updates(params, updates)
            params = jnp.clip(params, 0.01, 1.0)

            param_history.append(params)
            loss_history.append(loss)

            # Update the system:
            sys = sys.replace(
                dof_damping=params,
            )

        # Print the loss:
        print(f'Iteration: {i}, Loss: {loss}')

    print(f'Time taken: {time.time() - start_time}')

    # Run regressed params
    state = init_fn(sys, q_measured[0], qd_measured[0])
    q_history = []
    qd_history = []
    for i in range(control_history.shape[0]):
        state = step_fn(sys, state, control_history[i])
        q_history.append(state.q)
        qd_history.append(state.qd)

    # Run random trial with regressed params:
    q_test = q_measured[:, random_idx]
    qd_test = qd_measured[:, random_idx]
    ctrl_test = control_history[:, random_idx]
    state = jax.jit(pipeline.init)(sys, q_test[0], qd_test[0])
    q_history = []
    qd_history = []
    for i in range(control_history.shape[0]):
        state = jax.jit(pipeline.step)(sys, state, ctrl_test[i])
        q_history.append(state.q)
        qd_history.append(state.qd)

    param_history = np.asarray(param_history)
    loss_history = np.asarray(loss_history)
    q_history = np.asarray(q_history)
    qd_history = np.asarray(qd_history)
    q_history_init = np.asarray(q_history_init)
    qd_history_init = np.asarray(qd_history_init)

    fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5))
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

    fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5))
    fig.suptitle('Regressed Comparison')

    ax[0].plot(q_measured[:, random_idx, 0], color='orange', linestyle='--', linewidth=3.0)
    ax[0].plot(q_measured[:, random_idx, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    ax[0].plot(q_measured[:, random_idx, 2],  color='lightcoral', linestyle='--', linewidth=3.0)
    ax[0].plot(q_history[:, 0], color='orange', label='Abduction', linewidth=3.0)
    ax[0].plot(q_history[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    ax[0].plot(q_history[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    ax[0].legend()
    ax[0].set_title('Position')
    ax[0].set_ylabel('Position')

    ax[1].plot(qd_measured[:, random_idx, 0], color='orange', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_measured[:, random_idx, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_measured[:, random_idx, 2], color='lightcoral', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_history[:, 0], color='orange', label='Abduction', linewidth=3.0)
    ax[1].plot(qd_history[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    ax[1].plot(qd_history[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    ax[1].legend()
    ax[1].set_title('Velocity')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Velocity')

    plt.savefig('regressed.png')

    fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 5))
    fig.suptitle('Initial Comparison')

    ax[0].plot(q_measured[:, random_idx, 0], color='orange', linestyle='--', linewidth=3.0)
    ax[0].plot(q_measured[:, random_idx, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    ax[0].plot(q_measured[:, random_idx, 2],  color='lightcoral', linestyle='--', linewidth=3.0)
    ax[0].plot(q_history_init[:, 0], color='orange', label='Abduction', linewidth=3.0)
    ax[0].plot(q_history_init[:, 1], color='cornflowerblue', label='Hip', linewidth=3.0)
    ax[0].plot(q_history_init[:, 2], color='lightcoral', label='Knee', linewidth=3.0)
    ax[0].legend()
    ax[0].set_title('Position')
    ax[0].set_ylabel('Position')

    ax[1].plot(qd_measured[:, random_idx, 0], color='orange', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_measured[:, random_idx, 1], color='cornflowerblue', linestyle='--', linewidth=3.0)
    ax[1].plot(qd_measured[:, random_idx, 2], color='lightcoral', linestyle='--', linewidth=3.0)
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
