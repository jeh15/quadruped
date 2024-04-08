import os

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward
from brax.base import System
from brax.math import rotate, quat_inv

import jax
import jax.numpy as jnp

from ml_collections import config_dict

from control_utilities import remap_controller

# Configuration:
config = config_dict.ConfigDict()


class Unitree(PipelineEnv):

    def __init__(
        self,
        filename: str = 'unitree_a1/scene.xml',
        backend: str = 'mjx',
        params: config_dict.ConfigDict = config,
        **kwargs,
    ):
        filename = f'models/{filename}'
        self.filepath = os.path.join(
            os.path.dirname(
                os.path.dirname(__file__),
            ),
            filename,
        )
        sys = mjcf.load(self.filepath)

        # Set Simulation and Backend Parameters:
        self._dt = 0.01
        sys = sys.tree_replace({'opt.timestep': 0.001, 'dt': 0.001})

        n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend=backend, n_frames=n_frames)

        # Class Wide Parameters:

        # State indices to joints that can be actuated:
        self.motor_id = sys.actuator.qd_id

        # Default States:
        self.initial_q = jnp.array(
            [
                0, 0, 0.27, 1, 0, 0, 0,
                0, 0.9, -1.8,
                0, 0.9, -1.8,
                0, 0.9, -1.8,
                0, 0.9, -1.8,
            ]
        )
        self.base_control = jnp.array([
            0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8,
        ])

        # Remap Base Control for Model Input:
        self.action_range = jnp.tile(
            A=jnp.array([-1.0, 1.0]),
            reps=(self.action_size, 1),
        )
        self.control_range = self.sys.actuator_ctrlrange
        self.base_control_remap = remap_controller(
            jnp.expand_dims(self.base_control, axis=0),
            self.control_range,
            self.action_range,
        ).flatten()

        # State index ids:
        # x, y, z, quat
        self.body_id = jnp.array([0, 1, 2, 3, 4, 5, 6])
        self.base_x = jnp.array([0, 1, 2])
        self.base_w = jnp.array([3, 4, 5, 6])
        self.base_dw = jnp.array([3, 4, 5])
        # abduction, hip, knee
        self.front_right_x = jnp.array([7, 8, 9])
        self.front_right_dx = self.front_right_x - 1
        self.front_left_x = jnp.array([10, 11, 12])
        self.front_left_dx = self.front_left_x - 1
        self.back_right_x = jnp.array([13, 14, 15])
        self.back_right_dx = self.back_right_x - 1
        self.back_left_x = jnp.array([16, 17, 18])
        self.back_left_dx = self.back_left_x - 1

        # Parameters:
        self.q_size = sys.q_size()
        self.qd_size = sys.qd_size()
        self.calf_length = 0.2
        self.foot_radius = 0.025
        self.reset_noise = 0.05
        self.disturbance_range = 1.0
        self.push_range = 0.05
        self.kick_range = 1.0

        # Set Configuration:
        self.desired_orientation = jnp.array([1.0, 0.0, 0.0, 0.0])
        self.desired_height = 0.27
        self.min_z, self.max_z = 0.1, 0.4
        self.min_knee_z, self.max_knee_z = 0.05, 0.3

        # Command Tracking - Scaled with kernel function:
        self.forward_velocity_weight = 1.0 * self.dt
        self.turning_velocity_weight = 0.5 * self.dt

        # Penalize for large base velocities:
        self.linear_velocity_weight = 4.0 * self.dt
        self.angular_velocity_weight = 0.05 * self.dt

        # Base Height and Orientation:
        self.pose_weight = 1.0 * self.dt
        self.orientation_weight = 5.0 * self.dt

        # Feet and Joint Range:
        self.foot_height_weight = 0.0 * self.dt
        self.slip_weight = 1.0 * self.dt
        self.abduction_range_weight = 1.0 * self.dt
        self.hip_range_weight = 0.5 * self.dt
        self.knee_range_weight = 0.5 * self.dt

        # Regularization:
        self.reward_pose_regularization = 0.1 * self.dt
        self.velocity_regularization_weight = 1.0 * self.dt
        self.acceleration_regularization_weight = 0.001 * self.dt
        self.action_rate_weight = 1.0 * self.dt
        self.control_weight = 0.0005 * self.dt
        self.continuation_weight = 50.0 * self.dt
        self.termination_weight = -100.0 * self.dt

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, x_rng, y_rng, theta_rng, forward_rng, lateral_rng, turning_rng = jax.random.split(rng, 7)

        # Command:
        forward_command = jax.random.uniform(
            forward_rng,
            (),
            minval=-1,
            maxval=1,
        )
        lateral_command = jax.random.uniform(
            lateral_rng,
            (),
            minval=-0.3,
            maxval=0.3,
        )
        turning_command = jax.random.uniform(
            turning_rng,
            (),
            minval=-1.0,
            maxval=1.0,
        )
        command = jnp.array([forward_command, lateral_command, turning_command])

        # Random Noise:
        random_x = jax.random.uniform(
            x_rng,
            (),
            minval=-10,
            maxval=10,
        )
        random_y = jax.random.uniform(
            y_rng,
            (),
            minval=-10,
            maxval=10,
        )
        random_theta = jax.random.uniform(
            theta_rng,
            (),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )
        q_base_x = jnp.array(
            [random_x, random_y, self.initial_q[self.base_x][-1]],
        )
        q_base_w = jnp.array(
            [jnp.cos(random_theta / 2), 0, 0, jnp.sin(random_theta / 2)],
        )
        q_base = jnp.concatenate([q_base_x, q_base_w])
        q_joints = self.initial_q[7:]

        q = jnp.concatenate([q_base, q_joints])
        qd = jnp.zeros((self.qd_size,))

        pipeline_state = self.pipeline_init(q, qd)

        # Build model input:
        obs = self._get_observation(pipeline_state)
        state_input = jnp.concatenate([obs, self.base_control])
        model_input = jnp.tile(state_input, 5)
        model_input = jnp.concatenate([command, model_input])
        info = {
            'state_i': state_input,
            'state_i-1': state_input,
            'state_i-2': state_input,
            'state_i-3': state_input,
            'state_i-4': state_input,
            'model_input': model_input,
            'iteration': 1,
            'rng_key': rng,
            'command': command,
        }

        # Reward Function:
        reward = jnp.array([0.0])
        done = jnp.array(0, dtype=jnp.int64)
        zero = jnp.array([0.0])
        metrics = {
            'reward_forward_velocity': zero,
            'reward_turning_velocity': zero,
            'reward_linear_velocity': zero,
            'reward_angular_velocity': zero,
            'reward_action_rate': zero,
            'reward_pose': zero,
            'reward_orientation': zero,
            'reward_foot_height': jnp.zeros((4,)),
            'reward_slip': zero,
            'reward_abduction_range': zero,
            'reward_hip_range': jnp.zeros((4,)),
            'reward_knee_range': jnp.zeros((4,)),
            'reward_joint_regularization': zero,
            'reward_pose_regularization': zero,
            'reward_acceleration_regularization': zero,
            'reward_survival': zero,
            'previous_action': self.base_control,
            'previous_qd': jnp.zeros((12,)),
            'position': jnp.zeros((3,)),
            'orientation': jnp.zeros((4,)),
            'linear_velocity': jnp.zeros((3,)),
            'angular_velocity': jnp.zeros((3,)),
            'knee_termination': zero,
            'base_termination': zero,
        }

        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(
        self,
        state: State,
        action: jax.Array,
    ) -> State:
        # Helper Function:
        def kernel(x: jax.Array) -> jnp.ndarray:
            return jnp.exp(-jnp.linalg.norm(x) / 0.25)

        # RNG Keys:
        rng, magnitude_rng, angle_rng = jax.random.split(
            state.info['rng_key'], num=3,
        )

        # Disturbance:
        """
            Pushes every 10 steps -> 0.1 sec
            Kicks every 500 steps -> 5 sec
        """
        push_interval = 10
        kick_interval = 500
        # Only disturb in the xy directions:
        push_magnitude = jax.random.uniform(
            magnitude_rng,
            minval=-self.push_range,
            maxval=self.push_range,
        )
        kick_magnitude = jax.random.uniform(
            magnitude_rng,
            minval=-self.kick_range,
            maxval=self.kick_range,
        )
        disturbance_angle = jax.random.uniform(
            angle_rng,
            minval=-2*jnp.pi,
            maxval=2*jnp.pi,
        )
        push_mask = jnp.where(
            jnp.mod(state.info['iteration'], push_interval) == 0,
            1.0,
            0.0,
        )
        kick_mask = jnp.where(
            jnp.mod(state.info['iteration'], kick_interval) == 0,
            1.0,
            0.0,
        )
        push = jnp.array([
            jnp.cos(disturbance_angle) * push_magnitude,
            jnp.sin(disturbance_angle) * push_magnitude,
        ])
        kick = jnp.array([
            jnp.cos(disturbance_angle) * kick_magnitude,
            jnp.sin(disturbance_angle) * kick_magnitude,
        ])
        disturbance = push * push_mask + kick * kick_mask
        qvel = state.pipeline_state.qvel
        qvel = qvel.at[:2].set(disturbance + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        # Physics Step:
        pipeline_state = self.pipeline_step(
            state.pipeline_state,
            action,
        )

        # Observations and States:
        obs = self._get_observation(pipeline_state)
        q = pipeline_state.q
        qd = pipeline_state.qd
        q_joints = pipeline_state.q[7:]
        qd_joints = pipeline_state.qd[6:]

        # Foot Contact: (TODO: Change to air time)
        foot_positions = pipeline_state.site_xpos
        foot_contact = foot_positions[:, -1] - self.foot_radius
        contact = foot_contact < 1e-3
        reward_foot_height = jnp.where(
            contact,
            self.foot_height_weight,
            -self.foot_height_weight * jnp.linalg.norm(foot_positions[:, -1]),
        )

        # Foot Slip:
        idx = jnp.array([3, 6, 9, 12])
        position = pipeline_state.site_xpos
        feet_offset = position - pipeline_state.x.pos[idx]
        offset = base.Transform.create(pos=feet_offset)
        foot_velocity = offset.vmap().do(pipeline_state.xd.take(idx)).vel
        contact_mask = jnp.reshape(contact, (-1, 1))
        slip_error = foot_velocity[:, :2] * contact_mask
        reward_slip = -self.slip_weight * jnp.linalg.norm(slip_error)

        # Abduction Range:
        abduction_joint_idx = jnp.array([7, 10, 13, 16])
        abduction_q = q[abduction_joint_idx]
        reward_abduction_range = (
            -self.abduction_range_weight
            * jnp.linalg.norm(abduction_q)
        )

        # Hip Range:
        hip_joint_idx = jnp.array([8, 11, 14, 17])
        hip_range = jnp.pi / 2
        hip_ref = 0.9
        hip_error = hip_ref - q[hip_joint_idx]
        reward_hip_range = jnp.where(
            jnp.abs(hip_error) >= hip_range,
            -self.hip_range_weight,
            self.hip_range_weight,
        )

        # Knee Range:
        knee_joint_idx = jnp.array([9, 12, 15, 18])
        knee_ref = -1.8
        knee_range = jnp.pi / 6
        knee_q = q[knee_joint_idx]
        knee_error = knee_ref - knee_q
        reward_knee_range = jnp.where(
            jnp.abs(knee_error) >= knee_range,
            -self.knee_range_weight,
            self.knee_range_weight,
        )

        # Base Pose: Maintain Z Height
        base_x = q[self.base_x]
        pose_error = self.desired_height - base_x[-1]
        reward_pose = -self.pose_weight * jnp.linalg.norm(pose_error)

        # Barkour Formulation:
        base_w = q[self.base_w]
        up = jnp.array([0.0, 0.0, 1.0])
        rotation_error = rotate(up, base_w)
        reward_orientation = -self.orientation_weight * jnp.linalg.norm(
            rotation_error
        )

        # Velocity Tracking:
        linear_velocity = qd[self.base_x]
        angular_velocity = qd[self.base_dw]

        # Forward Velocity Tracking:
        base_frame_velocity = rotate(linear_velocity, quat_inv(base_w))
        linear_velocity_error = state.info['command'][:2] - base_frame_velocity[:2]
        forward_velocity_reward = self.forward_velocity_weight * kernel(
            linear_velocity_error,
        )

        # Turning Velocity Tracking:
        base_frame_angular_velocity = rotate(
            angular_velocity, quat_inv(base_w),
        )
        turning_velocity_error = state.info['command'][-1] - base_frame_angular_velocity[-1]
        turning_velocity_reward = self.turning_velocity_weight * kernel(
            turning_velocity_error,
        )

        # Base Regularization:
        reward_linear_velocity = -self.linear_velocity_weight * jnp.linalg.norm(
            linear_velocity[-1],
        )
        reward_angular_velocity = -self.angular_velocity_weight * jnp.linalg.norm(
            angular_velocity[:2],
        )

        # Control Regularization:
        action_rate = action - state.metrics['previous_action']
        reward_action_rate = -self.action_rate_weight * jnp.linalg.norm(
            action_rate
        )

        # Regularization:
        reward_joint_regularization = (
            -self.velocity_regularization_weight
            * jnp.linalg.norm(qd_joints)
        )
        qdd_joints = (qd_joints - state.metrics['previous_qd']) / self.dt
        reward_acceleration_regularization = (
            -self.acceleration_regularization_weight
            * jnp.linalg.norm(qdd_joints)
        )
        reward_pose_regularization = (
            -self.reward_pose_regularization * jnp.linalg.norm(
                q_joints - self.initial_q[7:]
            )
        )

        # Termination:
        base_termination = jnp.where(
            base_x[-1] < self.min_z, 1.0, 0.0,
        )
        base_termination = jnp.where(
            base_x[-1] > self.max_z, 1.0, base_termination,
        )
        knee_joint_pos_idx = jnp.array([3, 6, 9, 12])
        knee_x = pipeline_state.x.pos[knee_joint_pos_idx]
        knee_termination = jnp.where(
            knee_x[:, -1] < self.min_knee_z, 1.0, 0.0,
        )
        knee_termination = jnp.where(
            knee_x[:, -1] > self.max_knee_z, 1.0, knee_termination,
        )
        knee_termination = jnp.max(knee_termination)
        termination_cond = jnp.array([base_termination, knee_termination])
        termination = jnp.max(termination_cond)

        reward_survival = jnp.where(
            termination == 1.0,
            self.termination_weight,
            self.continuation_weight,
        )

        # Terminate flag:
        done = jnp.array(termination, dtype=jnp.int64)

        # Reward Function:
        reward = (
            reward_survival
            + forward_velocity_reward
            + turning_velocity_reward
            + reward_action_rate
            + reward_pose
            + reward_orientation
            + jnp.sum(reward_foot_height)
            + reward_slip
            + reward_linear_velocity
            + reward_angular_velocity
            + reward_pose_regularization
            + reward_joint_regularization
            + reward_acceleration_regularization
            + jnp.sum(reward_knee_range)
            + jnp.sum(reward_hip_range)
            + reward_abduction_range
        )
        reward = jnp.array([reward])
        reward = jnp.clip(reward, -1.0 * self.dt, 1000.0)
        metrics = {
            'reward_forward_velocity': jnp.array([forward_velocity_reward]),
            'reward_turning_velocity': jnp.array([turning_velocity_reward]),
            'reward_linear_velocity': jnp.array([reward_linear_velocity]),
            'reward_angular_velocity': jnp.array([reward_angular_velocity]),
            'reward_action_rate': jnp.array([reward_action_rate]),
            'reward_pose': jnp.array([reward_pose]),
            'reward_orientation': jnp.array([reward_orientation]),
            'reward_foot_height': reward_foot_height,
            'reward_slip': jnp.array([reward_slip]),
            'reward_abduction_range': jnp.array([reward_abduction_range]),
            'reward_hip_range': reward_hip_range,
            'reward_knee_range': reward_knee_range,
            'reward_pose_regularization': jnp.array([reward_pose_regularization]),
            'reward_joint_regularization': jnp.array([reward_joint_regularization]),
            'reward_acceleration_regularization': jnp.array([reward_acceleration_regularization]),
            'reward_survival': jnp.array([reward_survival]),
            'previous_action': action,
            'previous_qd': qd_joints,
            'position': base_x,
            'orientation': base_w,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'knee_termination': jnp.array([knee_termination]),
            'base_termination': jnp.array([base_termination]),
        }

        # Model input data:
        input_keys = ['state_i', 'state_i-1', 'state_i-2', 'state_i-3', 'state_i-4']
        state.info['state_i'] = jnp.concatenate([obs, action])
        state.info['state_i-1'] = state.info['state_i']
        state.info['state_i-2'] = state.info['state_i-1']
        state.info['state_i-3'] = state.info['state_i-2']
        state.info['state_i-4'] = state.info['state_i-3']
        model_input = [state.info[k] for k in state.info if k in input_keys]
        model_input = jnp.asarray(model_input).flatten()
        model_input = jnp.concatenate([state.info['command'], model_input])
        state.info['model_input'] = model_input
        state.info['rng_key'] = magnitude_rng
        state.info['iteration'] += 1

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    @property
    def action_size(self):
        return self.motor_id.shape[0]

    @property
    def observation_size(self):
        base_linear_velocity_size = 3
        base_angular_velocity_size = 3
        projected_gravity_size = 3
        joint_position_size = self.sys.q_size() - 7
        joint_velocity_size = self.sys.qd_size() - 6
        return (
            base_linear_velocity_size
            + base_angular_velocity_size
            + projected_gravity_size
            + joint_position_size
            + joint_velocity_size
        )

    def _get_observation(self, pipeline_state: base.State) -> jnp.ndarray:
        # Base Index:
        base_dx = jnp.array([0, 1, 2])
        base_dw = jnp.array([3, 4, 5])
        # Projected Gravity Calculation:
        base_inverse_quaternion = quat_inv(pipeline_state.x.rot[0])
        projected_gravity = rotate(
            jnp.array([0.0, 0.0, -1.0]), base_inverse_quaternion,
        )
        # Observations:
        base_height = jnp.array([pipeline_state.q[2]])
        base_linear_velocity = pipeline_state.qd[base_dx]
        base_angular_velocity = pipeline_state.qd[base_dw]
        joint_positions = pipeline_state.q[7:]
        joint_velocities = pipeline_state.qd[6:]

        observation = jnp.concatenate([
            base_height,
            base_linear_velocity,
            base_angular_velocity,
            projected_gravity,
            joint_positions,
            joint_velocities,
        ])
        return observation

    def _get_states(self, pipeline_state: base.State) -> jnp.ndarray:
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])

    def _get_body_position(self, pipeline_state: base.State) -> jnp.ndarray:
        return pipeline_state.x.pos[0]


# Utility Functions:
def get_point_taskspace_transform(
    vector: jax.Array,
    parent_x: jax.Array,
    parent_w: jax.Array,
    base: jax.Array,
) -> jnp.ndarray:
    # Rotate vector relative to base:
    vector_base = rotate(vector, base)

    # Rotate vector relative to its parent:
    vector_world = rotate(vector_base, parent_w) + parent_x

    return vector_world


def calculate_foot_position(
    sys: System,
    q: jax.Array,
    qd: jax.Array,
    calf_length: float,
) -> jnp.ndarray:
    # Calculate forward kinematics:
    x, _ = forward(sys, q, qd)
    base_w = x.rot[0]

    # Foot indices:
    idx = jnp.array([3, 6, 9, 12])
    parent_x = x.pos[idx, :]
    parent_w = x.rot[idx, :]

    # Calf length:
    vector = jnp.array([0.0, 0.0, -calf_length])

    # Taskspace transform for feet positions:
    foot_x = jax.vmap(
        get_point_taskspace_transform,
        in_axes=(None, 0, 0, None),
        out_axes=(0),
    )(vector, parent_x, parent_w, base_w)

    return foot_x
