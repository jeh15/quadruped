"""
    Unitree Go2 Environment:
        Height Control Environment for the Unitree Go2 quadruped robot.
"""

from typing import Any, Dict
from absl import app
import os

import flax.serialization
import jax
import jax.numpy as jnp

import numpy as np

import flax.struct
import flax.serialization

from brax import base
from brax import envs
from brax import math
from brax.base import System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html

import mujoco
from mujoco.mjx._src import math as mjx_math

from src.envs.utilities import collisions

# Types:
PRNGKey = jax.Array

@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_height: float = 1.0
    # Orientation Regularization Terms:
    angular_xy_velocity: float = -0.05
    orientation_regularization: float = -5.0
    pose_regularization: float = 0.1
    # Energy Regularization Terms:
    torque: float = -2e-4
    action_rate: float = -0.01
    mechanical_power: float = -1e-3
    acceleration: float = -1e-4
    # Auxilary Terms:
    termination: float = -1.0
    # Gait Terms:
    foot_slip: float = -0.1
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.05


@flax.struct.dataclass
class NoiseConfig:
    joint_position: float = 0.05
    joint_velocity: float = 1.5


@flax.struct.dataclass
class DisturbanceConfig:
    wait_times: list[float] = flax.struct.field(default_factory=lambda: [1.0, 3.0])
    durations: list[float] = flax.struct.field(default_factory=lambda: [0.05, 0.2])
    magnitudes: list[float] = flax.struct.field(default_factory=lambda: [0.0, 3.0])


def domain_randomize(sys: System, rng: PRNGKey) -> tuple[System, System]:
    @jax.vmap
    def randomize_parameters(rng):
        # Body IDs:
        FLOOR_BODY_ID = 0
        TORSO_BODY_ID = 1

        # Floor Friction:
        rng, key = jax.random.split(rng)
        geom_friction = jax.random.uniform(key, minval=0.4, maxval=1.0)
        friction = sys.geom_friction.at[FLOOR_BODY_ID, 0].set(geom_friction)

        # Joint Friction:
        rng, key = jax.random.split(rng)
        frictionloss = sys.dof_frictionloss[6:] * jax.random.uniform(
            key, shape=(12,), minval=0.9, maxval=1.1,
        )
        dof_frictionloss = sys.dof_frictionloss.at[6:].set(frictionloss)

        # Armature:
        rng, key = jax.random.split(rng)
        armature = sys.dof_armature[6:] * jax.random.uniform(
            key, shape=(12,), minval=1.0, maxval=1.05,
        )
        dof_armature = sys.dof_armature.at[6:].set(armature)

        # Center of Mass offset:
        rng, key = jax.random.split(rng)
        inertia_offset = jax.random.uniform(
            key, (3,), minval=-0.05, maxval=0.05,
        )
        body_ipos = sys.body_ipos.at[TORSO_BODY_ID].set(
            sys.body_ipos[TORSO_BODY_ID] + inertia_offset,
        )

        # Link mass randomization:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, (sys.nbody,), minval=0.9, maxval=1.1,
        )
        body_mass = sys.body_mass.at[:].set(sys.body_mass * delta)

        # Torso mass randomization:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, minval=-1.0, maxval=1.0,
        )
        body_mass = sys.body_mass.at[TORSO_BODY_ID].set(sys.body_mass[TORSO_BODY_ID] + delta)

        return (
            friction,
            dof_frictionloss,
            dof_armature,
            body_ipos,
            body_mass,
        )

    (
        friction,
        dof_frictionloss,
        dof_armature,
        body_ipos,
        body_mass,
    ) = randomize_parameters(rng)

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'dof_frictionloss': 0,
        'dof_armature': 0,
        'body_ipos': 0,
        'body_mass': 0,
    })

    sys = sys.tree_replace({
        'geom_friction': friction,
        'dof_frictionloss': dof_frictionloss,
        'dof_armature': dof_armature,
        'body_ipos': body_ipos,
        'body_mass': body_mass,
    })  # type: ignore

    return sys, in_axes


class UnitreeGo2Env(PipelineEnv):
    """Environment for training the Unitree Go2 quadruped joystick policy in MJX."""

    def __init__(
        self,
        filename: str = 'unitree_go2/scene_mjx_contact.xml',
        config: RewardConfig = RewardConfig(),
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        low_friction_model: bool = False,
        observation_model: str = 'default',
        **kwargs,
    ):
        filename = f'models/{filename}'
        self.filepath = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
            filename,
        )
        sys = mjcf.load(self.filepath)
        self.step_dt = 0.02
        sys = sys.tree_replace({'opt.timestep': 0.004})

        if low_friction_model:
            sys = sys.tree_replace({
                'dof_frictionloss': 0.01 * jnp.ones_like(sys.dof_frictionloss),
                'dof_armature': 0.005 * jnp.ones_like(sys.dof_armature),
            })

        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        self.kernel_sigma = config.kernel_sigma
        config_dict = flax.serialization.to_state_dict(config)
        del config_dict['kernel_sigma']
        self.reward_config = config_dict

        self.noise_config = NoiseConfig()
        self.disturbance_config = DisturbanceConfig()

        self.floor_geom_idx = self.sys.mj_model.geom('floor').id
        self.base_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
        )
        self.base_link_mass = self.sys.mj_model.body_subtreemass[self.base_idx]

        self.action_scale = action_scale
        self._kick_vel = kick_vel
        self.init_qd = jnp.zeros(sys.nv)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        self.initial_qpos = jnp.array([
            jnp.array(sys.mj_model.keyframe('home').qpos),
            jnp.array(sys.mj_model.keyframe('crouch').qpos),
            jnp.array(sys.mj_model.keyframe('prone_1').qpos),
            jnp.array(sys.mj_model.keyframe('prone_2').qpos),
            jnp.array(sys.mj_model.keyframe('prone_3').qpos),
            jnp.array(sys.mj_model.keyframe('tall_1').qpos),
            jnp.array(sys.mj_model.keyframe('tall_2').qpos),
        ])

        self.joint_lb = jnp.array([
            -1.0472, -1.5708, -2.7227,
            -1.0472, -1.5708, -2.7227,
            -1.0472, -0.5236, -2.7227,
            -1.0472, -0.5236, -2.7227,
        ])
        self.joint_ub = jnp.array([
            1.0472, 3.4907, -0.83776,
            1.0472, 3.4907, -0.83776,
            1.0472, 4.5379, -0.83776,
            1.0472, 4.5379, -0.83776,
        ])

        # Sites and Bodies:
        feet_geom = [
            'front_right',
            'front_left',
            'hind_right',
            'hind_left',
        ]
        feet_geom_idx = [
            self.sys.mj_model.geom(name).id for name in feet_geom
        ]
        assert not any(id_ == -1 for id_ in feet_geom_idx), 'Site not found.'
        self.feet_geom_idx = np.array(feet_geom_idx)
        feet_site = [
            'front_right_foot',
            'front_left_foot',
            'hind_right_foot',
            'hind_left_foot',
        ]
        feet_site_idx = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_idx), 'Site not found.'
        self.feet_site_idx = np.array(feet_site_idx)
        calf_body = [
            'front_right_calf',
            'front_left_calf',
            'hind_right_calf',
            'hind_left_calf',
        ]
        calf_body_idx = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, c)
            for c in calf_body
        ]
        assert not any(id_ == -1 for id_ in calf_body_idx), 'Body not found.'
        self.calf_body_idx = np.array(calf_body_idx)
        imu_site_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'imu'
        )
        assert not any(id_ == -1 for id_ in [imu_site_idx]), 'IMU site not found.'
        self.imu_site_idx = np.array(imu_site_idx)

        # Sensors:
        self.feet_position_sensor = [
            "fr_pos",
            "fl_pos",
            "hr_pos",
            "hl_pos",
        ]
        self.feet_linear_velocity_sensor = [
            "fr_global_linvel",
            "fl_global_linvel",
            "hr_global_linvel",
            "hl_global_linvel",
        ]

        # Constants:
        self.foot_radius = 0.022

        # Observation Size:
        if self.observation_model == 'default':
            self.num_observations = 37
        elif self.observation_model == 'gyroscope':
            self.num_observations = 40
        elif self.observation_model == 'gravity':
            self.num_observations = 40
        elif self.observation_model == 'gyroscope_gravity':
            self.num_observations = 43
        else:
            return NotImplementedError(
                f"Observation model {self.observation_model} not implemented."
            )
        
        self.num_privileged_observations = self.num_observations + 67

    def sample_command(self, rng: jax.Array) -> jax.Array:
        command_range = [0.078, 0.35]
        key, subkey = jax.random.split(rng)
        command = jax.random.uniform(
            subkey, shape=(1,), minval=command_range[0], maxval=command_range[1],
        )
        return command

    def reset(self, rng: PRNGKey) -> State:  # pytype: disable=signature-mismatch
        # Randomly sample the initial state:
        rng, key = jax.random.split(rng)
        qpos = jax.random.choice(
            key, self.initial_qpos,
        )

        # Initial Position:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, shape=(2,), minval=-0.5, maxval=0.5,
        )
        qpos = qpos.at[0:2].set(qpos[0:2] + delta)

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-jnp.pi, maxval=jnp.pi)
        rotation = mjx_math.axis_angle_to_quat(jnp.array([0, 0, 1]), yaw)
        quaternion = mjx_math.quat_mul(qpos[3:7], rotation)
        qpos = qpos.at[3:7].set(quaternion)

        # Initial Velocity:
        rng, key = jax.random.split(rng)
        qvel = self.init_qd.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        # Small Angle Deviation:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, shape=(12,), minval=-0.05, maxval=0.05,
        )
        qpos = qpos.at[7:].set(qpos[7:] + delta)

        # Small Velocity Deviation:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, shape=(12,), minval=-0.2, maxval=0.2,
        )
        qvel = qvel.at[6:].set(qvel[6:] + delta)

        pipeline_state = self.pipeline_init(qpos, qvel)

        # Disturbance: (Force Based)
        rng, disturbance_time_key, disturbance_duration_key, disturbance_magnitude_key = jax.random.split(rng, 4)
        time_until_next_disturbance = jax.random.uniform(
            disturbance_time_key,
            minval=self.disturbance_config.wait_times[0],
            maxval=self.disturbance_config.wait_times[1],
        )
        steps_until_next_disturbance = jnp.round(time_until_next_disturbance / self.dt).astype(
            jnp.int32
        )
        disturbance_duration = jax.random.uniform(
            disturbance_duration_key,
            minval=self.disturbance_config.durations[0],
            maxval=self.disturbance_config.durations[1],
        )
        disturbance_duration_steps = jnp.round(disturbance_duration / self.dt).astype(
            jnp.int32
        )
        disturbance_magnitude = jax.random.uniform(
            disturbance_magnitude_key,
            minval=self.disturbance_config.magnitudes[0],
            maxval=self.disturbance_config.magnitudes[1],
        )

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(12),
            'previous_velocity': jnp.zeros(12),
            'command': self.sample_command(key),
            'previous_contact': jnp.zeros(4, dtype=bool),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'steps_until_next_disturbance': steps_until_next_disturbance,
            'disturbance_duration': disturbance_duration,
            'disturbance_duration_steps': disturbance_duration_steps,
            'steps_since_last_disturbance': 0,
            'disturbance_step': 0,
            'disturbance_magnitude': disturbance_magnitude,
            'disturbance_direction': jnp.array([0.0, 0.0, 0.0]),
            'step': 0,
        }

        # Observation Tests:
        observation = self.get_observation(
            pipeline_state, state_info,
        )

        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        metrics = {'total_distance': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info,
        )
        return state

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info['rng'], 2)

        # Disturbance: (Force based)
        state = self.maybe_apply_perturbation(state)

        # Physics step:
        motor_targets = self.default_ctrl + action * self.action_scale
        pipeline_state = self.pipeline_step(
            state.pipeline_state, motor_targets,
        )

        # Observation data:
        observation = self.get_observation(
            pipeline_state,
            state.info,
        )
        joint_angles = pipeline_state.q[7:]
        joint_velocities = pipeline_state.qd[6:]

        # Foot contact data based on z-position:
        contact = jnp.array([
            collisions.geoms_colliding(pipeline_state, geom_id, self.floor_geom_idx)
            for geom_id in self.feet_geom_idx
        ])

        # Done if joint limits are reached or robot is falling:
        done = self.get_upvector(pipeline_state)[-1] < 0.0
        done |= jnp.any(joint_angles < self.joint_lb)
        done |= jnp.any(joint_angles > self.joint_ub)

        # Rewards:
        rewards = {
            'tracking_height': (
                self._reward_tracking_height(state.info['command'], pipeline_state.q[2])
            ),
            'angular_xy_velocity': self._reward_angular_velocity(
                self.get_global_angvel(pipeline_state),
            ),
            'orientation_regularization': self._reward_orientation_regularization(
                self.get_upvector(pipeline_state),
            ),
            'pose_regularization': self._reward_pose_regularization(
                joint_angles,
            ),
            'torque': self._reward_torques(pipeline_state.actuator_force),
            'action_rate': self._reward_action_rate(action, state.info['previous_action']),
            'mechanical_power': self._reward_mechanical_power(
                joint_velocities, pipeline_state.actuator_force,
            ),
            'acceleration': self._reward_acceleration(
                pipeline_state.qacc,
            ),
            'foot_slip': self._reward_foot_slip(
                pipeline_state, contact, state.info['command'],
            ),
            'termination': jnp.float64(
                self._reward_termination(done, state.info['step'])
            ) if jax.config.x64_enabled else jnp.float32(
                self._reward_termination(done, state.info['step'])
            ),
        }
        rewards = {
            k: v * self.reward_config[k] for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # State management
        state.info['previous_action'] = action
        state.info['previous_velocity'] = joint_velocities
        state.info['previous_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # Sample new command if more than 150 timesteps achieved
        state.info['command'] = jnp.where(
            state.info['step'] > 150,
            self.sample_command(cmd_rng),
            state.info['command'],
        )
        # Reset the step counter when done
        state.info['step'] = jnp.where(
            done | (state.info['step'] > 150), 0, state.info['step']
        )

        # Proxy Metrics:
        state.metrics['total_distance'] = math.normalize(
            pipeline_state.x.pos[self.base_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        state = state.replace(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
        )
        return state

    def get_observation(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> Dict[str, jax.Array]:
        """
            Observation: [
                relative_motor_positions,
                motor_velocities,
                previous_action,
                command,
            ]
        """
        q = pipeline_state.q[7:]
        qd = pipeline_state.qd[6:]

        # Gyroscope Noise:
        gyroscope = self.get_gyro(pipeline_state)
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        gyroscope_noise = jax.random.uniform(
            noise_key,
            shape=gyroscope.shape,
            minval=-self.noise_config.gyroscope,
            maxval=self.noise_config.gyroscope,
        )
        noisy_angular_rate = gyroscope + gyroscope_noise

        # Gravity noise:
        projected_gravity = self.get_gravity(pipeline_state)
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        gravity_noise = jax.random.uniform(
            noise_key,
            shape=projected_gravity.shape,
            minval=-self.noise_config.gravity_vector,
            maxval=self.noise_config.gravity_vector,
        )
        noisy_projected_gravity = projected_gravity + gravity_noise

        # Joint position noise:
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        joint_position_noise = jax.random.uniform(
            noise_key,
            shape=q.shape,
            minval=-self.noise_config.joint_position,
            maxval=self.noise_config.joint_position,
        )
        noisy_joint_positions = q + joint_position_noise

        # Joint velocity noise:
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        joint_velocity_noise = jax.random.uniform(
            noise_key,
            shape=qd.shape,
            minval=-self.noise_config.joint_velocity,
            maxval=self.noise_config.joint_velocity,
        )
        noisy_joint_velocities = qd + joint_velocity_noise

        if self.observation_model == 'default':
            observation = jnp.concatenate([
                noisy_joint_positions - self.default_pose,  # 12
                noisy_joint_velocities,                     # 12
                state_info['previous_action'],              # 12
                state_info['command'],                      # 1
            ])
            # Size: 37
        elif self.observation_model == 'gyroscope':
            observation = jnp.concatenate([
                noisy_angular_rate,                         # 3
                noisy_joint_positions - self.default_pose,  # 12
                noisy_joint_velocities,                     # 12
                state_info['previous_action'],              # 12
                state_info['command'],                      # 1
            ])
            # Size: 40
        elif self.observation_model == 'gravity':
            observation = jnp.concatenate([
                noisy_projected_gravity,                    # 3
                noisy_joint_positions - self.default_pose,  # 12
                noisy_joint_velocities,                     # 12
                state_info['previous_action'],              # 12
                state_info['command'],                      # 1
            ])
            # Size: 40
        elif self.observation_model == 'gyroscope_gravity':
            observation = jnp.concatenate([
                noisy_angular_rate,                         # 3s
                noisy_projected_gravity,                    # 3
                noisy_joint_positions - self.default_pose,  # 12
                noisy_joint_velocities,                     # 12
                state_info['previous_action'],              # 12
                state_info['command'],                      # 1
            ])
            # Size: 43
        else:
            return NotImplementedError(
                f"Observation model {self.observation_model} not implemented."
            )

        accelerometer = self.get_accelerometer(pipeline_state)
        linear_velocity = self.get_local_linvel(pipeline_state)
        global_angular_velocity = self.get_global_angvel(pipeline_state)
        actuator_force = pipeline_state.actuator_force
        feet_velocity = self.get_feet_velocity(pipeline_state).ravel()

        privileged_observation = jnp.concatenate([
            observation,                                                                                # 37, 40, or 43
            accelerometer,                                                                              # 3
            gyroscope,                                                                                  # 3
            projected_gravity,                                                                          # 3
            linear_velocity,                                                                            # 3
            global_angular_velocity,                                                                    # 3
            q - self.default_pose,                                                                      # 12
            qd,                                                                                         # 12
            actuator_force,                                                                             # 12
            feet_velocity,                                                                              # 12
            pipeline_state.xfrc_applied[self.base_idx, :3],                                             # 3
            jnp.asarray([
                state_info['steps_since_last_disturbance'] >= state_info['steps_until_next_disturbance']
            ]),                                                                                         # 1
        ])
        # Size (Privileged Observations - 67): 104, 107, or 110

        return {
            'state': observation,
            'privileged_state': privileged_observation,
        }

    def _reward_tracking_height(
        self, command: jax.Array, global_base_z: jax.Array
    ) -> jax.Array:
        # Tracking of height command (z axis)
        error = jnp.sum(jnp.square(command - global_base_z))
        return jnp.exp(-error / self.kernel_sigma)

    def _reward_angular_velocity(
        self, global_base_angvel: jax.Array,
    ) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(global_base_angvel[:2]))

    def _reward_orientation_regularization(
        self, base_z_axis: jax.Array,
    ) -> jax.Array:
        # Penalize non flat base orientation
        return jnp.sum(jnp.square(base_z_axis[:2]))
    
    def _reward_pose_regularization(
        self, qpos: jax.Array,
    ) -> jax.Array:
        weight = jnp.array([1.0, 1.0, 0.1] * 4)
        error = jnp.sum(jnp.square(qpos - self.default_pose) * weight)
        return jnp.exp(-error)

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(
        self, action: jax.Array, previous_action: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(action - previous_action))

    def _reward_mechanical_power(
        self, qd: jax.Array, torques: jax.Array
    ) -> jax.Array:
        # Penalize mechanical power
        return jnp.sum(jnp.abs(torques) * jnp.abs(qd))

    def _reward_acceleration(
        self, qacc: jax.Array,
    ) -> jax.Array:
        # Penalize Motor/Joint Acceleration
        return jnp.sqrt(jnp.sum(jnp.square(qacc)))

    def _reward_foot_slip(
        self,
        pipeline_state: base.State,
        contact: jax.Array,
        commands: jax.Array,
    ) -> jax.Array:
        feet_vel = self.get_feet_velocity(pipeline_state)
        vel_xy = feet_vel[..., :2]
        vel_xy_sq = jnp.sum(jnp.square(vel_xy), axis=-1)
        return jnp.sum(vel_xy_sq * contact)

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 150)

    @staticmethod
    def get_sensor_data(
        model: mujoco.MjModel, pipeline_state: base.State, sensor_name: str
    ) -> jax.Array:
        """Gets sensor data given sensor name."""
        sensor_id = model.sensor(sensor_name).id
        sensor_adr = model.sensor_adr[sensor_id]
        sensor_dim = model.sensor_dim[sensor_id]
        return pipeline_state.sensordata[sensor_adr : sensor_adr + sensor_dim]

    def get_upvector(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(self.sys.mj_model, pipeline_state, "upvector")

    def get_gravity(self, pipeline_state: base.State) -> jax.Array:
        return pipeline_state.site_xmat[self.imu_site_idx].T @ jnp.array([0, 0, -1])

    def get_global_linvel(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "global_linvel"
        )

    def get_global_angvel(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "global_angvel"
        )

    def get_local_linvel(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "local_linvel"
        )

    def get_accelerometer(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(
            self.sys.mj_model, pipeline_state, "imu_acceleration"
        )

    def get_gyro(self, pipeline_state: base.State) -> jax.Array:
        return self.get_sensor_data(self.sys.mj_model, pipeline_state, "imu_gyro")

    def get_feet_pos(self, pipeline_state: base.State) -> jax.Array:
        return jnp.vstack([
            self.get_sensor_data(self.sys.mj_model, pipeline_state, sensor_name)
            for sensor_name in self.feet_position_sensor
        ])

    def get_feet_velocity(self, pipeline_state: base.State) -> jax.Array:
        return jnp.vstack([
            self.get_sensor_data(self.sys.mj_model, pipeline_state, sensor_name)
            for sensor_name in self.feet_linear_velocity_sensor
        ])

    # Adapted from mujoco_playground:
    def maybe_apply_perturbation(self, state: State) -> State:
        def gen_dir(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jnp.pi * 2)
            return jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])

        def apply_perturbation(state: State) -> State:
            t = state.info["disturbance_step"] * self.dt
            u_t = 0.5 * jnp.sin(jnp.pi * t / state.info["disturbance_duration"])
            # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
            force = (
                u_t  # (unitless)
                * self.base_link_mass  # kg
                * state.info["disturbance_magnitude"]  # m/s
                / state.info["disturbance_duration"]  # 1/s
            )
            xfrc_applied = jnp.zeros((self.sys.nbody, 6))
            xfrc_applied = xfrc_applied.at[self.base_idx, :3].set(
                force * state.info["disturbance_direction"]
            )
            pipeline_state = state.pipeline_state.replace(xfrc_applied=xfrc_applied)
            state = state.replace(pipeline_state=pipeline_state)
            state.info["steps_since_last_disturbance"] = jnp.where(
                state.info["disturbance_step"] >= state.info["disturbance_duration_steps"],
                0,
                state.info["steps_since_last_disturbance"],
            )
            state.info["disturbance_step"] += 1
            return state

        def wait(state: State) -> State:
            state.info["rng"], rng = jax.random.split(state.info["rng"])
            state.info["steps_since_last_disturbance"] += 1
            xfrc_applied = jnp.zeros((self.sys.mj_model.nbody, 6))
            pipeline_state = state.pipeline_state.replace(xfrc_applied=xfrc_applied)
            state.info["disturbance_step"] = jnp.where(
                state.info["steps_since_last_disturbance"]
                >= state.info["steps_until_next_disturbance"],
                0,
                state.info["disturbance_step"],
            )
            state.info["disturbance_direction"] = jnp.where(
                state.info["steps_since_last_disturbance"]
                >= state.info["steps_until_next_disturbance"],
                gen_dir(rng),
                state.info["disturbance_direction"],
            )
            return state.replace(pipeline_state=pipeline_state)

        return jax.lax.cond(
            state.info["steps_since_last_disturbance"]
            >= state.info["steps_until_next_disturbance"],
            apply_perturbation,
            wait,
            state,
        )

    def np_observation(
        self,
        mj_data: mujoco.MjData,
        command: np.ndarray,
        previous_action: np.ndarray,
        add_noise: bool = True,
    ) -> np.ndarray:
        # Numpy implementation of the observation function:
        q = mj_data.qpos[7:]
        qd = mj_data.qvel[6:]

        if add_noise:
            q = q + np.random.uniform(
                low=-self.noise_config.joint_position,
                high=self.noise_config.joint_position,
                size=q.shape,
            )
            qd = qd + np.random.uniform(
                low=-self.noise_config.joint_velocity,
                high=self.noise_config.joint_velocity,
                size=qd.shape,
            )

        observation = np.concatenate([
            q - self.default_ctrl,
            qd,
            previous_action,
            command,
        ])

        return {
            'state': observation,
            'privileged_state': np.zeros((self.num_privileged_observations,)),
        }
    
    def hardware_observation(
        self,
        imu_state: Any,
        motor_state: Any,
        command: np.ndarray,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        # Numpy implementation of the observation function:
        # Set to Correct Data Type:
        joint_positions = np.asarray(motor_state.q, dtype=np.float32)
        joint_velocities = np.asarray(motor_state.qd, dtype=np.float32)

        # Cast to float64:
        joint_positions = joint_positions.astype(np.float64)
        joint_velocities = joint_velocities.astype(np.float64)

        observation = np.concatenate([
            joint_positions - self.default_ctrl,
            joint_velocities,
            previous_action,
            command,
        ])

        return {
            'state': observation,
            'privileged_state': np.zeros((self.num_privileged_observations,)),
        }

envs.register_environment('unitree_go2', UnitreeGo2Env)


def main(argv=None):
    env = UnitreeGo2Env()
    rng = jax.random.PRNGKey(0)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    state = reset_fn(rng)

    num_steps = 100
    states = []
    for i in range(num_steps):
        print(f"Step: {i}")
        state = step_fn(state, jnp.zeros_like(env.default_ctrl))
        print(state.obs['state'].shape)
        print(state.obs['privileged_state'].shape)
        states.append(state.pipeline_state)

    html_string = html.render(
        sys=env.sys.tree_replace({'opt.timestep': env.step_dt}),
        states=states,
        height="100vh",
        colab=False,
    )
    html_path = os.path.join(
        os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                ),
            ),
        ),
        "visualization/visualization.html",
    )

    with open(html_path, "w") as f:
        f.writelines(html_string)


if __name__ == '__main__':
    app.run(main)
