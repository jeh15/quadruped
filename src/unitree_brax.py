import os

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.kinematics import forward
from brax.base import System, Motion, Transform
from brax.math import rotate, quat_inv, normalize
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

        # Reward Weights:
        self.reward_weights = {
            'linear_velocity_tracking': 2.0 * self.dt,
            'angular_velocity_tracking': 2.0 * self.dt,
            'base_height_regularization': 2.0 * self.dt,
            'foot_contact': 1.0 * self.dt,
            'orientation': -0.1 * self.dt,
            'linear_velocity_regularization': -0.1 * self.dt,
            'angular_velocity_regularization': -0.05 * self.dt,
            'joint_motion': -0.001 * self.dt,
            'action_rate': -0.25 * self.dt,
            'pose_regularization': -0.5 * self.dt,
            'abduction_regularization': -0.5 * self.dt,
            'hip_regularization': -0.5 * self.dt,
            'knee_regularization': -0.5 * self.dt,
            'foot_slip': -0.1 * self.dt,
            'termination': 1.0 * self.dt,
        }

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
        qd = jnp.zeros((self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_states(pipeline_state)

        # Reward Function:
        reward = jnp.array(0.0)
        done = jnp.array(0, dtype=jnp.int64)
        zero = jnp.array([0.0])

        metrics = {
            'rewards': {k: 0.0 for k in self.reward_weights.keys()},
            'position': jnp.zeros((3,)),
            'orientation': jnp.zeros((4,)),
            'linear_velocity': jnp.zeros((3,)),
            'angular_velocity': jnp.zeros((3,)),
            'knee_termination': zero,
            'base_termination': zero,
        }

        info = {
            'command': jnp.zeros(3),
            'previous_action': jnp.zeros(self.action_size),
            'qd_joints_previous': jnp.zeros_like(q_joints),
            'feet_air_time': jnp.zeros(4),
            'last_contact': jnp.zeros(4, dtype=jnp.bool),
        }

        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(
        self,
        state: State,
        action: jax.Array,
    ) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(
            state.pipeline_state,
            action,
        )

        # States
        obs = self._get_states(pipeline_state)
        x = pipeline_state.x
        dx = pipeline_state.xd
        q_joints = pipeline_state.q[7:]
        qd_joints = pipeline_state.qd[6:]

        # Foot Contact:
        foot_positions = pipeline_state.site_xpos
        foot_contact = foot_positions[:, -1] - self.foot_radius
        contact = foot_contact < 1e-3
        contact_filter_mm = contact | state.info['last_contact']
        contact_filter_cm = (foot_contact < 3e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filter_mm
        state.info['feet_air_time'] += self.dt

        # Termination:
        base_x = pipeline_state.x.pos[0]
        base_w = pipeline_state.x.rot[0]
        linear_velocity = pipeline_state.xd.vel[0]
        angular_velocity = pipeline_state.xd.ang[0]
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

        # Terminate flag:
        done = jnp.array(termination, dtype=jnp.int64)

        # Reward Function:
        rewards = {
            'linear_velocity_tracking': (
                self.reward_linear_velocity_tracking(
                    state.info['command'], x, dx,
                )
            ),
            'angular_velocity_tracking': (
                self.reward_angular_velocity_tracking(
                    state.info['command'], x, dx,
                )
            ),
            'linear_velocity_regularization': (
                self.reward_linear_velocity_z(dx)
            ),
            'angular_velocity_regularization': (
                self.reward_angular_velocity_xy(dx)
            ),
            'orientation': self.reward_orientation(x),
            'joint_motion': self.reward_joint_motion(
                qd_joints, state.info['qd_joints_previous'],
            ),
            'action_rate': self.reward_action_rate(
                action, state.info['previous_action'],
            ),
            'pose_regularization': self.reward_pose_regularization(q_joints),
            'base_height_regularization': self.reward_base_height_regularization(x),
            'abduction_regularization': self.reward_abduction_regularization(
                q_joints,
            ),
            'hip_regularization': self.reward_hip_regularization(q_joints),
            'knee_regularization': self.reward_knee_regularization(q_joints),
            'foot_contact': self.reward_foot_contact(contact),
            'foot_slip': self.reward_foot_slip(pipeline_state, contact_filter_cm),
            'termination': self.reward_survival(termination),
        }

        rewards = {
            k: v * self.reward_weights[k] for k, v in rewards.items()
        }
        reward = sum(rewards.values())

        metrics = {
            'rewards': rewards,
            'position': base_x,
            'orientation': base_w,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'knee_termination': jnp.array([knee_termination]),
            'base_termination': jnp.array([base_termination]),
        }

        state.info['feet_air_time'] *= ~contact_filter_mm
        state.info['previous_action'] = action
        state.info['qd_joints_previous'] = qd_joints
        state.info['last_contact'] = contact

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

    def reward_linear_velocity_z(self, dx: Motion) -> jax.Array:
        return jnp.square(dx.vel[0, -1])

    def reward_angular_velocity_xy(self, dx: Motion) -> jax.Array:
        return jnp.linalg.norm(dx.ang[0, :2])

    def reward_orientation(self, x: Transform) -> jax.Array:
        oreientation_error = 1 - jnp.square(
            jnp.dot(x.rot[0], self.desired_orientation),
        )
        return oreientation_error

    def reward_joint_acceleration(
        self,
        qd_joints: jax.Array,
        qd_joints_previous: jax.Array,
    ) -> jax.Array:
        return jnp.linalg.norm((qd_joints - qd_joints_previous) / self.dt)

    def reward_joint_motion(
        self,
        qd_joints: jax.Array,
        qd_joints_previous: jax.Array,
    ) -> jax.Array:
        joint_velocity = jnp.linalg.norm(qd_joints)
        joint_acceleration = jnp.linalg.norm(
            (qd_joints - qd_joints_previous) / self.dt,
        )
        return joint_velocity + joint_acceleration

    def reward_action_rate(
        self,
        action: jax.Array,
        previous_action: jax.Array,
    ) -> jax.Array:
        return jnp.linalg.norm(action - previous_action)

    def reward_linear_velocity_tracking(
        self,
        command: jax.Array,
        x: Transform,
        dx: Motion,
    ) -> jax.Array:
        body_velocity = rotate(dx.vel[0], quat_inv(x.rot[0]))
        velocity_error = jnp.linalg.norm(command[:2] - body_velocity[:2])
        return self.phi(velocity_error)

    def reward_angular_velocity_tracking(
        self,
        command: jax.Array,
        x: Transform,
        dx: Motion,
    ) -> jax.Array:
        body_angular_velocity = rotate(dx.ang[0], quat_inv(x.rot[0]))
        angular_velocity_error = jnp.linalg.norm(
            command[-1] - body_angular_velocity[-1],
        )
        return self.phi(angular_velocity_error)

    def reward_feet_air_time(
        self,
        command: jax.Array,
        feet_air_time: jax.Array,
        contact: jax.Array,
    ) -> jax.Array:
        reward_air_time = jnp.sum((feet_air_time - 0.5) * contact)
        reward_air_time *= (
            normalize(command[:2])[-1] > 0.05
        )
        return reward_air_time

    def reward_foot_contact(
        self,
        contact: jax.Array,
    ) -> jax.Array:
        reward_contact = jnp.where(
            contact,
            1.0,
            -1.0,
        )
        return jnp.linalg.norm(reward_contact)

    def reward_pose_regularization(self, q_joints: jax.Array) -> jax.Array:
        return jnp.linalg.norm(q_joints - self.initial_q[7:])

    def reward_base_height_regularization(self, x: Transform) -> jax.Array:
        height_error = jnp.linalg.norm(x.pos[0, -1] - self.desired_height)
        return self.phi(height_error)

    def reward_abduction_regularization(self, q_joints: jax.Array) -> jax.Array:
        idx = jnp.array([0, 3, 6, 9])
        abduction_ref = 0.0
        return jnp.linalg.norm(q_joints[idx] - abduction_ref)

    def reward_hip_regularization(self, q_joints: jax.Array) -> jax.Array:
        idx = jnp.array([1, 4, 7, 10])
        hip_range = jnp.pi / 2
        hip_ref = 0.9
        hip_error = hip_ref - q_joints[idx]
        reward_hip = jnp.where(
            jnp.abs(hip_error) >= hip_range,
            -1.0,
            1.0,
        )
        return jnp.linalg.norm(reward_hip)

    def reward_knee_regularization(self, q_joints: jax.Array) -> jax.Array:
        idx = jnp.array([2, 5, 8, 11])
        knee_range = jnp.pi / 6
        knee_ref = -1.8
        knee_error = knee_ref - q_joints[idx]
        reward_knee = jnp.where(
            jnp.abs(knee_error) >= knee_range,
            -1.0,
            1.0,
        )
        return jnp.linalg.norm(reward_knee)

    def reward_pose_zero_command(
        self,
        commands: jax.Array,
        q_joints: jax.Array,
    ) -> jax.Array:
        return jnp.linalg.norm(q_joints - self.initial_q[7:]) * (
            normalize(commands[:2])[-1] < 0.1
        )

    def reward_foot_slip(
        self,
        pipeline_state: base.State,
        contact_mask: jax.Array,
    ) -> jax.Array:
        idx = jnp.array([3, 6, 9, 12])
        position = pipeline_state.site_xpos
        feet_offset = position - pipeline_state.x.pos[idx]
        offset = base.Transform.create(pos=feet_offset)
        foot_velocity = offset.vmap().do(pipeline_state.xd.take(idx)).vel
        contact_mask = jnp.reshape(contact_mask, (-1, 1))
        return jnp.linalg.norm(foot_velocity[:, :2] * contact_mask)

    def reward_survival(self, termination: jax.Array) -> jax.Array:
        return jnp.where(
            termination == 1.0,
            -1.0,
            1.0,
        )

    @staticmethod
    def phi(x: jax.Array) -> jnp.ndarray:
        return jnp.exp(-x / 0.25)


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
