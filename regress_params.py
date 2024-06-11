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
from brax.io import mjcf, html
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

    initial_params = sys.dof_damping
    friction_param = sys.dof_damping
    target_param = 5.2  # 0.5239

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
    q_batch = jnp.asarray(np.swapaxes(q_batch, 1, 2))
    qd_batch = jnp.asarray(np.swapaxes(qd_batch, 1, 2))
    ctrl_batch = jnp.asarray(np.swapaxes(ctrl_batch, 1, 2))

    # Shuffle Data:
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    # VMAP and Jit the pipeline functions:
    vmap_init_fn = jax.vmap(pipeline.init, in_axes=(None, 0, 0))
    init_fn = jax.jit(vmap_init_fn)

    vmap_step_fn = jax.vmap(pipeline.step, in_axes=(None, 0, 0))
    step_fn = jax.jit(vmap_step_fn)

    # Initialize the parameter and optimizer:
    num_learning_iterations = 50
    transition_steps = num_learning_iterations * minibatch_size
    schedule_fn = optax.polynomial_schedule(
        init_value=1e-2,
        end_value=1e-4,
        power=2,
        transition_steps=transition_steps,
    )
    solver = optax.adam(learning_rate=1e-2)
    params = jnp.array(friction_param)
    opt_state = solver.init(params)

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

    param_history = []
    loss_history = []
    start_time = time.time()

    def update_solver(gradient, opt_state, params):
        updates, opt_state = solver.update(
            gradient, opt_state, params,
        )
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 0.01, 100.0)
        return params, opt_state

    def inner_loop(carry, xs):
        sys, opt_state, params = carry
        batch = xs

        states = init_fn(
            sys,
            batch.q[:, 0],
            batch.qd[:, 0],
        )

        loss, grad = grad_fn(
            sys, states, batch,
        )

        # Extract the gradients:
        gradient = grad.dof_damping

        # Update the parameters:
        params, opt_state = update_solver(
            gradient, opt_state, params,
        )

        # Update the system:
        sys = sys.replace(
            dof_damping=params,
        )

        return (sys, opt_state, params), (loss, params)

    def outer_loop(carry, unused_t, data):
        sys, opt_state, params, key = carry

        # Shuffle data:
        key, subkey = jax.random.split(key)
        q = jax.random.permutation(subkey, data.q, axis=0)
        qd = jax.random.permutation(subkey, data.qd, axis=0)
        ctrl = jax.random.permutation(subkey, data.ctrl, axis=0)
        ctrl = jnp.swapaxes(ctrl, 1, 2)
        shuffled_data = jax.tree.map(
            lambda x, y, z: minibatch(x, y, z), q, qd, ctrl,
        )

        (sys, opt_state, params), (loss, param_history) = jax.lax.scan(
            f=inner_loop,
            init=(sys, opt_state, params),
            xs=shuffled_data,
            length=num_batches,
        )

        return (sys, opt_state, params, subkey), (loss, param_history)

    # Training Loop:
    data = minibatch(q_batch, qd_batch, ctrl_batch)
    (sys, opt_state, params, _), (loss_history, param_history) = jax.lax.scan(
        f=functools.partial(outer_loop, data=data),
        init=(sys, opt_state, params, subkey),
        xs=(),
        length=num_learning_iterations,
    )
    loss_history = np.asarray(loss_history.flatten())
    param_history = np.reshape(param_history, (-1, 3))

    print(f'Time taken: {time.time() - start_time}')

    # Run initial comparison of random trial:
    sys = sys.replace(
        dof_damping=initial_params,
    )
    random_idx = jax.random.randint(subkey, (), minval=0, maxval=num_trials-1)
    q_initial_trial = q_measured[:, random_idx]
    qd_initial_trial = qd_measured[:, random_idx]
    ctrl_initial_trial = control_history[:, random_idx]
    state = jax.jit(pipeline.init)(sys, q_initial_trial[0], qd_initial_trial[0])
    q_history_init = []
    qd_history_init = []
    prior_state_history = []
    for i in range(control_history.shape[0]):
        state = jax.jit(pipeline.step)(sys, state, ctrl_initial_trial[i])
        prior_state_history.append(state)
        q_history_init.append(state.q)
        qd_history_init.append(state.qd)

    # Run random trial with regressed params:
    sys = sys.replace(
        dof_damping=params,
    )
    q_test = q_measured[:, random_idx]
    qd_test = qd_measured[:, random_idx]
    ctrl_test = control_history[:, random_idx]
    state = jax.jit(pipeline.init)(sys, q_test[0], qd_test[0])
    q_history = []
    qd_history = []
    posterior_state_history = []
    for i in range(control_history.shape[0]):
        state = jax.jit(pipeline.step)(sys, state, ctrl_test[i])
        posterior_state_history.append(state)
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

    plt.savefig('regress_params.pdf')

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

    plt.savefig('regressed.pdf')

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

    plt.savefig('initial.pdf')

    # Generate Visualizations:
    html_string = html.render(
        sys,
        prior_state_history,
        height="100vh",
        colab=False,
    )
    html_path = os.path.join(
        os.path.dirname(__file__),
        "visualization/prior.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)

    html_string = html.render(
        sys,
        posterior_state_history,
        height="100vh",
        colab=False,
    )

    html_path = os.path.join(
        os.path.dirname(__file__),
        "visualization/posterior.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
