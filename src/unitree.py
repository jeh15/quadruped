import os

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward
from brax.base import System
from brax.math import rotate
from brax.actuator import to_tau

import jax
import jax.numpy as jnp

from ml_collections import config_dict

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

        # Set Backend Parameters:
        sys = sys.replace(dt=0.001)
        # Control at 100 Hz -> n_frames * dt
        physics_steps_per_control_step = 10
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = backend

        super().__init__(sys, **kwargs)

        # Class Wide Parameters:
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

        # State indices to joints that can be actuated:
        self.motor_id = sys.actuator.qd_id

        # Parameters:
        self.q_size = sys.q_size()
        self.qd_size = sys.qd_size()
        self.calf_length = 0.2
        self.foot_radius = 0.025
        self.reset_noise = 0.05

        # Set Configuration:
        self.desired_orientation = jnp.array([1.0, 0.0, 0.0, 0.0])
        self.desired_height = 0.27
        self.min_z, self.max_z = 0.1, 0.4
        self.min_knee_z, self.max_knee_z = 0.05, 0.3

        # Scaled with Phi function:
        self.pose_weight = 2.0 * self.dt
        self.orientation_weight = 1.0 * self.dt
        self.linear_velocity_weight = 1.0 * self.dt
        self.angular_velocity_weight = 1.0 * self.dt

        self.foot_height_weight = 5.0 * self.dt
        self.abduction_range_weight = 0.5 * self.dt
        self.hip_range_weight = 0.5 * self.dt
        self.knee_range_weight = 0.5 * self.dt

        self.reward_pose_regularization = 0.01 * self.dt
        self.velocity_regularization_weight = 0.1 * self.dt
        self.acceleration_regularization_weight = 0.001 * self.dt
        self.action_rate_weight = 0.25 * self.dt
        self.control_weight = 0.0005 * self.dt
        self.continuation_weight = 10.0 * self.dt
        self.termination_weight = -10.0 * self.dt

        # Unused
        self.linear_velocity_regularization = 0.1 * self.dt
        self.angular_velocity_regularization = 0.1 * self.dt

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, q_rng, qd_rng = jax.random.split(rng, 3)

        # Random Noise:
        # low, high = -self.reset_noise, self.reset_noise
        # q_base = self.initial_q[self.body_id]
        # q_joints = self.initial_q[7:] + jax.random.uniform(
        #     q_rng,
        #     (self.sys.q_size() - 7,),
        #     minval=low,
        #     maxval=high,
        # )
        # q = jnp.concatenate([q_base, q_joints])
        # qd = 0.1 * high * jax.random.normal(qd_rng, (self.sys.qd_size(),))

        # No Random Noise:
        q = self.initial_q
        qd = jnp.zeros((self.qd_size,))

        pipeline_state = self.pipeline_init(q, qd)

        # Build model input:
        obs = self._get_states(pipeline_state)
        action = jnp.zeros((self.action_size,))
        current_input = jnp.concatenate([obs, action])
        info = {
            'state_i': current_input,
            'state_i-1': jnp.zeros_like(current_input),
            'state_i-2': jnp.zeros_like(current_input),
            'state_i-3': jnp.zeros_like(current_input),
            'state_i-4': jnp.zeros_like(current_input),
            'model_input': jnp.zeros(5 * current_input.shape[0]),
        }

        # Reward Function:
        reward = jnp.array([0.0])
        done = jnp.array(0, dtype=jnp.int64)
        zero = jnp.array([0.0])
        metrics = {
            'reward_linear_velocity': zero,
            'reward_angular_velocity': zero,
            'reward_action_rate': zero,
            'reward_pose': zero,
            'reward_orientation': zero,
            'reward_foot_height': jnp.zeros((4,)),
            'reward_abduction_range': zero,
            'reward_hip_range': jnp.zeros((4,)),
            'reward_knee_range': jnp.zeros((4,)),
            'reward_joint_regularization': zero,
            'reward_pose_regularization': zero,
            'reward_acceleration_regularization': zero,
            'reward_survival': zero,
            'previous_action': jnp.zeros((12,)),
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
        def phi(x: jax.Array) -> jnp.ndarray:
            return jnp.exp(-jnp.linalg.norm(x) / 0.25)

        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(
            state.pipeline_state,
            action,
        )

        # Observations and States:
        obs = self._get_states(pipeline_state)
        q = pipeline_state.q
        qd = pipeline_state.qd
        q_joints = pipeline_state.q[7:]
        qd_joints = pipeline_state.qd[6:]

        # Foot Contact:
        foot_positions = pipeline_state.site_xpos
        foot_contact = foot_positions[:, -1] - self.foot_radius
        contact = foot_contact < 1e-3
        reward_foot_height = jnp.where(
            contact,
            self.foot_height_weight,
            -self.foot_height_weight * foot_positions[:, -1],
        )

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
            -self.hip_range_weight,
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
        reward_pose = self.pose_weight * phi(pose_error)

        # Base Orientation:
        base_w = q[self.base_w]
        orientation_error = 1 - jnp.square(
            jnp.dot(base_w, self.desired_orientation),
        )
        reward_orientation = -self.orientation_weight * orientation_error

        # Velocity Tracking:
        # TODO(jeh15): Change to tracking a desired velocity vector -> norm(vel_des - vel)
        linear_velocity = qd[self.base_x]
        angular_velocity = qd[self.base_dw]
        reward_linear_velocity = self.linear_velocity_weight * phi(
            linear_velocity,
        )
        reward_angular_velocity = self.angular_velocity_weight * phi(
            angular_velocity,
        )

        # Control regularization:
        reward_action_rate = -self.action_rate_weight * jnp.linalg.norm(
            action - state.metrics['previous_action'],
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
            + reward_action_rate
            + reward_pose
            + reward_orientation
            + jnp.sum(reward_foot_height)
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
        metrics = {
            'reward_linear_velocity': jnp.array([reward_linear_velocity]),
            'reward_angular_velocity': jnp.array([reward_angular_velocity]),
            'reward_action_rate': jnp.array([reward_action_rate]),
            'reward_pose': jnp.array([reward_pose]),
            'reward_orientation': jnp.array([reward_orientation]),
            'reward_foot_height': reward_foot_height,
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
        state.info['model_input'] = model_input

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
        return self.sys.q_size() + self.sys.qd_size()

    @property
    def step_dt(self):
        return self.dt * self._n_frames

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