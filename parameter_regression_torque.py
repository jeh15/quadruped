from typing import Tuple

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

import time

# Scipy Filter:
from scipy import signal
from scipy.signal import butter, sosfilt, convolve



import matplotlib.pyplot as plt


jax.config.update('jax_enable_x64', True)

# jax.config.update('jax_disable_jit', True)


@flax.struct.dataclass
class minibatch:
    q: jax.Array
    qd: jax.Array
    torque: jax.Array
    ctrl: jax.Array


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, btype='low', analog=False, output='sos', fs=fs)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    sos = butter_lowpass(cutoff, fs, order=order)
    y = sosfilt(sos, data)
    return y


def main(argv=None):
    filename = 'models/unitree_go2/go2_regression_model.xml'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    sys = mjcf.load(filepath)
    sys = sys.tree_replace({'opt.timestep': 0.004})

    control_rate = 0.02
    control_steps = int(control_rate / sys.opt.timestep)

    file_path = os.path.join(
        os.path.dirname(__file__),
        'data/unitree_data_1.pkl',
    )
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    data = np.asarray(data)
    
    # Get the number of time steps and trials
    num_motors = 12
    num_dof = 3
    num_trials, num_time_steps, _ = data.shape

    # Parse Data:
    q_measured = data[:, :, :12]
    qd_measured = data[:, :, 12:24]
    torque_measured = data[:, :, 24:36]
    setpoints = data[:, :, 36:]

    # Filter Velocity Data: (Window of 25 Still captures the initial value)
    qd_filtered = []
    window = signal.windows.hann(25)
    for data in qd_measured:
        y = []
        for i in range(num_motors):
            y.append(convolve(data[:, i], window, mode='same') / sum(window)) 
        y = np.asarray(y).swapaxes(0, 1)
        qd_filtered.append(y)

    qd_filtered = np.asarray(qd_filtered)

    # Concatenate Legs into different trials:
    process_fn = lambda x: np.concatenate(
            np.split(
                np.reshape(x, (num_trials, num_time_steps, 4, 3)),
                indices_or_sections=4,
                axis=2,
            ),
            axis=0,
        ).squeeze()

    q_measured = process_fn(q_measured)
    qd_measured = process_fn(qd_measured)
    qd_filtered = process_fn(qd_filtered)
    torque_measured = process_fn(torque_measured)
    setpoints = process_fn(setpoints)

    # Structure Data into Minibatches:
    minibatch_size = 25
    num_batches = num_time_steps // minibatch_size

    # Batch: (4 * num_trial, num_batches, minibatch_size, num_dof)
    q_batch = np.reshape(
        q_measured, (4 * num_trials, num_batches, minibatch_size, num_dof),
    )
    qd_batch = np.reshape(
        qd_measured, (4 * num_trials, num_batches, minibatch_size, num_dof),
    )
    torque_batch = np.reshape(
        torque_measured, (4 * num_trials, num_batches, minibatch_size, num_dof),
    )
    ctrl_batch = np.reshape(
        setpoints, (4 * num_trials, num_batches, minibatch_size, num_dof),
    )

    # Axis order: (batch, trial, minibatch, num_dof)
    q_batch = jnp.asarray(np.swapaxes(q_batch, 0, 1))
    qd_batch = jnp.asarray(np.swapaxes(qd_batch, 0, 1))
    torque_batch = jnp.asarray(np.swapaxes(torque_batch, 0, 1))
    ctrl_batch = jnp.asarray(np.swapaxes(ctrl_batch, 0, 1))

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
    solver = optax.adam(learning_rate=1e-2)

    # Parameters to Regress:
    damping_params = sys.dof_damping
    kp_params = sys.actuator_gainprm[:, 0]

    params = {
        'dof_damping': damping_params,
        'kp': kp_params,
    }

    opt_state = solver.init(params)


    def unroll(
        system: brax.System,
        state: pipeline.State,
        control: jnp.ndarray,
    ) -> pipeline.State:
        def f(carry, unused_t):
            state = carry
            state = step_fn(system, state, control)
            return state, state
        
        final_state, _ = jax.lax.scan(
            f=f,
            init=state,
            length=control_steps,
        )

        return final_state

    def loss_function(
        system: brax.System,
        state: pipeline.State,
        batch: minibatch,
        minibatch_size: int,
    ) -> Tuple[pipeline.State, Tuple[jnp.ndarray, jnp.ndarray]]:
        def scan_fn(carry, data):
            state = carry
            control = data
            state = unroll(system, state, control)
            
            return state, (state.q, state.qd, state.actuator_force)

        _, (q, qd, torque) = jax.lax.scan(
            f=scan_fn,
            init=state,
            xs=batch.ctrl,
            length=minibatch_size,
        )

        # Reshape the data: axis -> (trials, time, num_dof)
        q = jnp.swapaxes(q, 0, 1)
        qd = jnp.swapaxes(qd, 0, 1)
        torque = jnp.swapaxes(torque, 0, 1)

        rmse_fn = lambda x, y: jnp.sqrt(jnp.mean(jnp.square(x - y)))
        
        weights = {
            'position': 1.0,
            'velocity': 0.1,
            'torque': 1.0,
        }

        losses = {
            'position': rmse_fn(q, batch.q),
            'velocity': rmse_fn(qd, batch.qd),
            'torque': rmse_fn(torque, batch.torque),
        }

        losses = {
            k: v * weights[k] for k, v in losses.items()
        }

        loss = sum(losses.values())

        return loss

    loss_fn = functools.partial(loss_function, minibatch_size=minibatch_size)
    grad_function = jax.value_and_grad(loss_fn, allow_int=True)
    grad_fn = jax.jit(grad_function)

    param_history = []
    loss_history = []

    def update_solver(gradient, opt_state, params):
        updates, opt_state = solver.update(
            gradient, opt_state, params,
        )
        params = optax.apply_updates(params, updates)
        params = jax.tree.map(lambda x: jnp.clip(x, 0.01, 100.0), params)
        return params, opt_state

    def inner_loop(carry, xs):
        sys, opt_state, params = carry
        batch = xs

        states = init_fn(
            sys,
            batch.q[:, 0],
            batch.qd[:, 0],
        )

        # # TODO(jeh15): Stabilize Simulation?
        # num_stabilization_iterations = 5
        # for _ in range(num_stabilization_iterations):
        #     states = step_fn(sys, states, batch.q[:, 0])

        loss, grad = grad_fn(
            sys, states, batch,
        )

        # Extract the gradients:
        gradient = {
            'dof_damping': grad.dof_damping,
            'kp': grad.actuator_gainprm[:, 0]
        }

        # Update the parameters:
        params, opt_state = update_solver(
            gradient, opt_state, params,
        )

        # Update the system:
        dof_damping = params['dof_damping']
        kp = params['kp']

        gain = sys.actuator_gainprm.at[:, 0].set(kp)
        bias = sys.actuator_biasprm.at[:, 0].set(-kp)

        # Update Motor Stiffness:
        sys = sys.replace(
            actuator_gainprm=gain,
            actuator_biasprm=bias,
        )

        # Update Motor Stiffness and Damping:
        # sys = sys.replace(
        #     dof_damping=dof_damping,
        #     actuator_gainprm=gain,
        #     actuator_biasprm=bias,
        # )

        return (sys, opt_state, params), (loss, params)

    def outer_loop(carry, unused_t, data):
        sys, opt_state, params, key = carry

        # Shuffle data:
        key, subkey = jax.random.split(key)
        q = jax.random.permutation(subkey, data.q, axis=0)
        qd = jax.random.permutation(subkey, data.qd, axis=0)
        torque = jax.random.permutation(subkey, data.torque, axis=0)
        ctrl = jax.random.permutation(subkey, data.ctrl, axis=0)
        ctrl = jnp.swapaxes(ctrl, 1, 2)
        shuffled_data = jax.tree.map(
            lambda w, x, y, z: minibatch(w, x, y, z),
            q, qd, torque, ctrl,
        )

        (sys, opt_state, params), (loss, param_history) = jax.lax.scan(
            f=inner_loop,
            init=(sys, opt_state, params),
            xs=shuffled_data,
            length=num_batches,
        )

        return (sys, opt_state, params, subkey), (loss, param_history)

    # Training Loop:
    data = minibatch(q_batch, qd_batch, torque_batch, ctrl_batch)
    start_time = time.time()
    (sys, opt_state, params, _), (loss_history, param_history) = jax.lax.scan(
        f=functools.partial(outer_loop, data=data),
        init=(sys, opt_state, params, subkey),
        xs=(),
        length=num_learning_iterations,
    )
    print(f'Time taken: {time.time() - start_time}')
    loss_history = np.asarray(loss_history.flatten())

    # Save Regression:
    data_directory = os.path.join(
        os.path.dirname(__file__),
        'data',
    )
    os.makedirs(data_directory, exist_ok=True)
    param_file = os.path.join(
        data_directory,
        'param_regression_history_torque.pkl',
    )
    loss_file = os.path.join(
        data_directory,
        'loss_history_torque.pkl',
    )

    with open(param_file, 'wb') as f:
        pickle.dump(param_history, f)

    with open(loss_file, 'wb') as f:
        pickle.dump(loss_history, f)


if __name__ == '__main__':
    app.run(main)
