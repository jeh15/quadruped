from absl import app
import os
import functools
import time
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from brax.io import mjcf

from unitree_api_bindings import unitree_api


jax.config.update('jax_enable_x64', True)


def main(argv=None):
    filename = 'models/unitree_go2/scene_mjx_fixed.xml'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    sys = mjcf.load(filepath)

    # Generate Control Trajectory:
    def random_initial_state(
        home_position: jax.Array,
        key: jax.Array,
    ) -> jnp.ndarray:
        home_position = jnp.reshape(home_position, (4, -1))
        abduction_range = jnp.array([
            -0.2, 0.2,
        ])
        hip_range = jnp.array([
            -0.5, 0.5,
        ])
        knee_range = jnp.array([
            -0.5, 0.5,
        ])
        bounds = jnp.vstack(
            (abduction_range, hip_range, knee_range),
        )

        qpos = home_position + jax.random.uniform(
            key, shape=home_position.shape, minval=bounds[:, 0], maxval=bounds[:, 1],
        )
        return qpos
    
    def control_sweep(
        key: jax.Array,
        num_time_steps: int = 500,
    ) -> jnp.ndarray:
        key, abduction_offset_key, front_hip_offset_key, hind_hip_offset_key, knee_offset_key = jax.random.split(key, 5)
        key, abduction_amplitude_key, hip_amplitude_key, knee_amplitude_key = jax.random.split(key, 4)
        key, abduction_frequency_key, hip_frequency_key, knee_frequency_key = jax.random.split(key, 4)

        abduction_offset = [-0.6, 0.3]
        front_hip_offset = [-0.72, 0.9]
        hind_hip_offset = [0.9, 1.6]
        knee_offset = [-1.9, -1.6]

        abduction_offset = jax.random.uniform(
            abduction_offset_key, shape=(4,), minval=abduction_offset[0], maxval=abduction_offset[1],
        )
        front_hip_offset = jax.random.uniform(
            front_hip_offset_key, shape=(2,), minval=front_hip_offset[0], maxval=front_hip_offset[1],
        )
        hind_hip_offset = jax.random.uniform(
            hind_hip_offset_key, shape=(2,), minval=hind_hip_offset[0], maxval=hind_hip_offset[1],
        )
        knee_offset = jax.random.uniform(
            knee_offset_key, shape=(4,), minval=knee_offset[0], maxval=knee_offset[1],
        )
        hip_offset = jnp.concatenate([front_hip_offset, hind_hip_offset])

        abduction_amplitude = jax.random.uniform(
            abduction_amplitude_key, shape=(4,), minval=-0.3, maxval=0.3,
        )
        hip_amplitude = jax.random.uniform(
            hip_amplitude_key, shape=(4,), minval=-0.85, maxval=0.85,
        )
        knee_amplitude = jax.random.uniform(
            knee_amplitude_key, shape=(4,), minval=-0.75, maxval=0.75,
        )

        abduction_frequency = jax.random.randint(
            abduction_frequency_key, shape=(4,), minval=15, maxval=150,
        )
        hip_frequency = jax.random.randint(
            hip_frequency_key, shape=(4,), minval=15, maxval=150,
        )
        knee_frequency = jax.random.randint(
            knee_frequency_key, shape=(4,), minval=15, maxval=150,
        )

        x = jnp.arange(num_time_steps)

        sinusoid_fn = jax.jit(
            jax.vmap(
                lambda x, y, z, w: x + y * jnp.sin(w / z), 
                in_axes=(None, None, None, 0),
                out_axes=0,
            ),
        )

        abduction_trajectory = sinusoid_fn(
            abduction_offset, abduction_amplitude, abduction_frequency, x,
        )
        hip_trajectory = sinusoid_fn(
            hip_offset, hip_amplitude, hip_frequency, x,
        )
        knee_trajectory = sinusoid_fn(
            knee_offset, knee_amplitude, knee_frequency, x,
        )

        format_array = lambda x: jnp.expand_dims(x, -1)
        abduction_trajectory = format_array(abduction_trajectory)
        hip_trajectory = format_array(hip_trajectory)
        knee_trajectory = format_array(knee_trajectory)

        control_trajectory = jnp.concatenate(
            (abduction_trajectory, hip_trajectory, knee_trajectory),
            axis=-1,
        )
        return control_trajectory
        

    vmap_initial_state = jax.vmap(
        random_initial_state, in_axes=(None, 0), out_axes=0,
    )
    initial_state_fn = jax.jit(vmap_initial_state)

    random_control_fn = functools.partial(
        random_control,
        num_time_steps=1000,
    )
    vmap_control_trajectory = jax.vmap(
        random_control_fn, in_axes=0, out_axes=0,
    )
    control_trajectory_fn = jax.jit(vmap_control_trajectory)

    key = jax.random.key(42)
    key, state_key, ctrl_key = jax.random.split(key, 3)
    num_trials = 50
    state_keys = jax.random.split(state_key, num_trials)
    control_keys = jax.random.split(ctrl_key, num_trials)

    home_position = jnp.array(sys.mj_model.keyframe('home').qpos[:])
    qpos = initial_state_fn(home_position, state_keys)

    control_trajectories = control_trajectory_fn(
        control_keys,
    )
    control_trajectories = np.asarray(control_trajectories)
    
    # Initalize Unitree API:
    control_rate = 0.02
    control_rate_ns = 2e7

    # Initialize Unitree-Api:
    network_name = "enx7cc2c647de4f"
    inner_control_rate = 2000
    unitree_driver = unitree_api.UnitreeDriver(
        network_name,
        inner_control_rate,
    )
    unitree_driver.initialize()

    # Default Control:
    motor_commands = unitree_api.MotorCommand()
    motor_commands.q_setpoint = [0.0, 0.9, -1.8] * 4
    motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
    motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
    motor_commands.stiffness = [0.0, 0.0, 0.0] * 4
    motor_commands.damping = [0.0, 0.0, 0.0] * 4
    unitree_driver.update_command(motor_commands)

    print('Press any key to start get up sequence...')
    input()

    # Initialize Thread:
    unitree_driver.initialize_thread()

    # Ramp to Default Control:
    ramp_time = 5.0
    num_steps = 1000
    stifness_ramp = np.linspace(0.0, 60.0, num_steps)
    damping_ramp = np.linspace(0.0, 5.0, num_steps)
    for stiffness, damping in zip(stifness_ramp, damping_ramp):
        motor_commands.stiffness = [stiffness, stiffness, stiffness] * 4
        motor_commands.damping = [damping, damping, damping] * 4
        unitree_driver.update_command(motor_commands)
        time.sleep(ramp_time / num_steps)

    print('Press any key to start tests...')
    input()

    # Control Loop:
    data = []
    for trial, control_trajectory in enumerate(control_trajectories):
        print(f'Trial {trial + 1} of {num_trials}...')

        # Move to initial position for test:
        motor_state = unitree_driver.get_motor_state()

        desired_position = control_trajectory[0].flatten()
        current_position = np.asarray(motor_state.q)

        # Linear Iterpolation to desired position:
        trajectory = np.linspace(
            current_position, desired_position, num=200,
        )
        next_time_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        for position in trajectory:
            next_time_ns += control_rate_ns
            motor_commands.q_setpoint = position
            motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
            motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
            motor_commands.stiffness = [60.0, 60.0, 60.0] * 4
            motor_commands.damping = [5.0, 5.0, 5.0] * 4
            unitree_driver.update_command(motor_commands)
            now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            if now_ns < next_time_ns:
                sleep_time_ns = next_time_ns - now_ns
                time.sleep(sleep_time_ns / 1e9)
            else:
                print('Warning: Control rate exceeded.')
                next_time_ns = now_ns

        pass

        # Run Test:
        motor_states = []
        previous_setpoint = position.flatten()
        for setpoint in control_trajectory:
            next_time_ns += control_rate_ns
            
            # This will be the output of the previous step:
            motor_state = unitree_driver.get_motor_state()
            joint_positions = np.asarray(motor_state.q)
            joint_velocities = np.asarray(motor_state.qd)
            joint_torques = np.asarray(motor_state.torque_estimate)
            joint_state = np.concatenate(
                (joint_positions, joint_velocities, joint_torques, previous_setpoint),
                axis=0,
            )
            motor_states.append(joint_state)
            previous_setpoint = setpoint.flatten()

            motor_commands.q_setpoint = setpoint.flatten()
            motor_commands.qd_setpoint = [0.0, 0.0, 0.0] * 4
            motor_commands.torque_feedforward = [0.0, 0.0, 0.0] * 4
            motor_commands.stiffness = [35.0, 35.0, 35.0] * 4
            motor_commands.damping = [0.5, 0.5, 0.5] * 4
            unitree_driver.update_command(motor_commands)

            now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            if now_ns < next_time_ns:
                sleep_time_ns = next_time_ns - now_ns
                time.sleep(sleep_time_ns / 1e9)
            else:
                print('Warning: Control rate exceeded.')
                next_time_ns = now_ns

        data.append(motor_states)

    # Save Data:
    data = np.asarray(data)

    # Save data to file:
    data_directory = os.path.join(
        os.path.dirname(__file__),
        'data',
    )
    os.makedirs(data_directory, exist_ok=True)
    data_file = os.path.join(
        data_directory,
        'unitree_data_1.pkl',
    )
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)

    # Stop Thread:
    unitree_driver.stop_thread()



if __name__ == '__main__':
    app.run(main)