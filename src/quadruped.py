import os

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward
from brax.base import System
from brax.math import rotate

import jax
import jax.numpy as jnp

from ml_collections import config_dict

# Configuration:
config = config_dict.ConfigDict()
config.forward_reward_weight = 1.25
config.ctrl_cost_weight = 0.1
config.healthy_reward = 5.0
config.terminate_when_unhealthy = True
config.healthy_z_range = (0.05, 0.4)
config.reset_noise_scale = 1e-2
config.exclude_current_positions_from_observation = True


class Quadruped(PipelineEnv):

    def __init__(
        self,
        filename: str = 'quadruped_brax.xml',
        backend: str = 'generalized',
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
                0, 0, 0.2, 1.0, 0, 0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
            ],
        )

        # State index ids:
        # x, y, z, quat
        self.body_id = jnp.array([0, 1, 2, 3, 4, 5, 6])
        self.base_x = jnp.array([0, 1, 2])
        self.base_w = jnp.array([3, 4, 5, 6])
        self.base_dw = jnp.array([3, 4, 5])
        # hip, knee
        self.front_left_id = jnp.array([7, 8])
        self.front_right_id = jnp.array([9, 10])
        self.back_left_id = jnp.array([11, 12])
        self.back_right_id = jnp.array([13, 14])

        # State indices to joints that can be actuated:
        self.motor_id = sys.actuator.qd_id

        # Parameters:
        self.q_size = 15
        self.qd_size = self.q_size - 1
        self.calf_length = 0.17
        self.reset_noise = 0.1

        # Set Configuration:
        self.desired_orientation = jnp.array([1.0, 0.0, 0.0, 0.0])
        self.desired_height = 0.2
        self.min_z, self.max_z = 0.125, 0.275

        self.foot_height_weight = 2.0 * sys.dt
        self.pose_weight = 1.0 * sys.dt
        self.orientation_weight = 1.0 * sys.dt
        self.linear_velocity_weight = 1.0 * sys.dt
        self.angular_velocity_weight = 0.5 * sys.dt
        self.linear_velocity_regularization = 4.0 * sys.dt
        self.angular_velocity_regularization = 0.05 * sys.dt
        self.control_weight = 0.1 * sys.dt
        self.continuation_weight = 1.0 * sys.dt

        self._forward_reward_weight = params.forward_reward_weight
        self._ctrl_cost_weight = params.ctrl_cost_weight
        self._healthy_reward = params.healthy_reward
        self._terminate_when_unhealthy = params.terminate_when_unhealthy
        self._healthy_z_range = params.healthy_z_range
        self._reset_noise_scale = params.reset_noise_scale
        self._exclude_current_positions_from_observation = (
            params.exclude_current_positions_from_observation
        )

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, q_rng, qd_rng = jax.random.split(rng, 3)

        # Random Noise:
        low, high = -self.reset_noise, self.reset_noise
        q_base = self.initial_q[self.body_id]
        q_joints = self.initial_q[7:] + jax.random.uniform(
            q_rng,
            (self.sys.q_size() - 7,),
            minval=low,
            maxval=high,
        )
        q = jnp.concatenate([q_base, q_joints])
        qd = high * jax.random.normal(qd_rng, (self.sys.qd_size(),))

        # # No Random Noise:
        # q = self.initial_q
        # qd = jnp.zeros((self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)

        obs = self._get_states(pipeline_state)

        # Reward Function:
        reward = jnp.array([0.0])
        done = jnp.array(0, dtype=jnp.int64)
        zero = jnp.array([0.0])
        metrics = {
            'reward_linear_velocity': zero,
            'reward_angular_velocity': zero,
            'reward_control': zero,
            'reward_pose': zero,
            'reward_orientation': zero,
            'reward_foot_height': jnp.zeros((4,)),
            'reward_duty_cycle': zero,
            'reward_survival': zero,
            'position': jnp.zeros((3,)),
            'orientation': jnp.zeros((4,)),
            'linear_velocity': jnp.zeros((3,)),
            'angular_velocity': jnp.zeros((3,)),
        }

        return State(pipeline_state, obs, reward, done, metrics)

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
        obs = self._get_states(pipeline_state)

        # Foot Positions:
        foot_positions = calculate_foot_position(
            self.sys,
            pipeline_state.q,
            pipeline_state.qd,
        )
        foot_z = foot_positions[:, -1]
        foot_padding = 0.025
        reward_foot_height = jnp.where(
            jnp.abs(foot_z) <= foot_padding,
            self.foot_height_weight,
            -self.foot_height_weight * jnp.square(foot_z),
        )

        # Base Pose: Maintain Z Height
        base_x = pipeline_state.q[self.base_x]
        pose_error = self.desired_height - base_x[-1]
        reward_pose = self.pose_weight * phi(pose_error)

        # Base Orientation:
        base_w = pipeline_state.q[self.base_w]
        orientation_error = 1 - jnp.square(jnp.dot(base_w, self.desired_orientation))
        reward_orientation = self.orientation_weight * phi(orientation_error)

        # Velocity Tracking:
        # Based on the magnitude of the velocity.
        # TODO(jeh15): Change to tracking a desired velocity vector -> norm(vel_des - vel)
        linear_velocity = pipeline_state.qd[self.base_x]
        angular_velocity = pipeline_state.qd[self.base_dw]
        reward_linear_velocity = self.linear_velocity_weight * phi(
            linear_velocity,
        )
        reward_angular_velocity = self.angular_velocity_weight * phi(
            angular_velocity,
        )

        # Control regularization:
        reward_control = -self.control_weight * jnp.sum(jnp.square(action))

        # Termination:
        base_x = pipeline_state.q[self.base_x]
        termination = jnp.where(
            base_x[-1] < self.min_z, 1.0, 0.0,
        )
        termination = jnp.where(
            base_x[-1] > self.max_z, 1.0, termination,
        )
        reward_survival = (1.0 - termination) * self.continuation_weight

        # Terminate flag:
        done = jnp.array(termination, dtype=jnp.int64)

        # Reward Function:
        reward = (
            reward_survival
            + reward_control
            + reward_pose
            + reward_orientation
            + jnp.sum(reward_foot_height)
            + reward_linear_velocity
            + reward_angular_velocity
        )
        reward = jnp.array([reward])
        zero = jnp.array([0.0])
        metrics = {
            'reward_linear_velocity': jnp.array([reward_linear_velocity]),
            'reward_angular_velocity': jnp.array([reward_angular_velocity]),
            'reward_control': jnp.array([reward_control]),
            'reward_pose': jnp.array([reward_pose]),
            'reward_orientation': jnp.array([reward_orientation]),
            'reward_foot_height': reward_foot_height,
            'reward_duty_cycle': zero,
            'reward_survival': jnp.array([reward_survival]),
            'position': base_x,
            'orientation': base_w,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
        }

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
) -> jnp.ndarray:
    # Calculate forward kinematics:
    x, _ = forward(sys, q, qd)
    base_w = x.rot[0]

    # Foot indices:
    idx = jnp.array([2, 4, 6, 8])
    parent_x = x.pos[idx, :]
    parent_w = x.rot[idx, :]

    # Calf length:
    vector = jnp.array([0.17, 0.0, 0.0])

    # Taskspace transform for feet positions:
    foot_x = jax.vmap(
        get_point_taskspace_transform,
        in_axes=(None, 0, 0, None),
        out_axes=(0),
    )(vector, parent_x, parent_w, base_w)

    return foot_x
