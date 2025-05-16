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

from src.envs.utilities import collisions


# Types:
PRNGKey = jax.Array


@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_height: float = 2.0
    # Orientation Regularization Terms:
    tracking_height_error: float = -5.0
    linear_xy_velocity: float = -1.0
    angular_xy_velocity: float = -0.05
    orientation_regularization: float = -1.0
    pose_regularization: float = -0.1
    # Energy Regularization Terms:
    torque: float = -2e-4
    action_rate: float = -0.01
    acceleration: float = -1e-4
    # Foot Contact Terms:
    foot_contact: float = -0.1
    foot_slip: float = -0.1
    # Auxilary Terms:
    termination: float = -1.0
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.1


@flax.struct.dataclass
class NoiseConfig:
    joint_position: float = 0.05
    joint_velocity: float = 1.5
    gyroscope: float = 0.2
    gravity_vector: float = 0.05


def domain_randomize(sys: System, rng: PRNGKey) -> tuple[System, System]:
    @jax.vmap
    def randomize_parameters(rng):
        # Body IDs:
        FLOOR_BODY_ID = 0
        TORSO_BODY_ID = 1

        # Floor Friction:
        rng, key = jax.random.split(rng)
        geom_friction = jax.random.uniform(key, minval=0.6, maxval=1.0)
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
        filename: str = 'unitree_go2/scene_mjx_collision.xml',
        config: RewardConfig = RewardConfig(),
        noise_config: NoiseConfig = NoiseConfig(),
        action_scale: float = 0.3,
        low_friction_model: bool = False,
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
                'dof_frictionloss': 0.0 * jnp.ones_like(sys.dof_frictionloss),
                'dof_armature': 0.005 * jnp.ones_like(sys.dof_armature),
            })

        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        # Configurations:
        self.kernel_sigma = config.kernel_sigma
        config_dict = flax.serialization.to_state_dict(config)
        del config_dict['kernel_sigma']
        self.reward_config = config_dict
        self.noise_config = noise_config
        self.action_scale = action_scale

        # Floor and Base IDs:
        self.floor_geom_idx = self.sys.mj_model.geom('floor').id
        self.base_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
        )
        self.base_link_mass = self.sys.mj_model.body_subtreemass[self.base_idx]

        # Initial and Default States:
        self.init_qd = jnp.zeros(sys.nv)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        self.initial_qpos = jnp.array([
            jnp.array(sys.mj_model.keyframe('home').qpos),
            jnp.array(sys.mj_model.keyframe('prone_1').qpos),
            jnp.array(sys.mj_model.keyframe('prone_2').qpos),
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

        # Observation Size:
        self.num_observations = 43
        self.num_privileged_observations = self.num_observations + 49

    def sample_command(self, rng: jax.Array) -> jax.Array:
        key, subkey = jax.random.split(rng)
        command_range = [0.1, 0.35]
        command = jax.random.uniform(
            subkey,
            shape=(1,),
            minval=command_range[0],
            maxval=command_range[1],
        )
        return command

    def reset(self, rng: PRNGKey) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        # Initial Position:
        qpos = jax.random.choice(
            key, self.initial_qpos,
        )

        # Initial Velocity:
        rng, key = jax.random.split(rng)
        qvel = self.init_qd.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.1, maxval=0.1)
        )

        # Small Velocity Deviation:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, shape=(12,), minval=-0.1, maxval=0.1,
        )
        qvel = qvel.at[6:].set(qvel[6:] + delta)

        pipeline_state = self.pipeline_init(qpos, qvel)

        # Get the initial position of the IMU:
        imu_qpos = pipeline_state.site_xpos[self.imu_site_idx]

        # Command Sampling:
        rng, command_interval_key, command_sample_key = jax.random.split(
            rng, 3,
        )
        time_until_next_command = jax.random.uniform(
            command_interval_key, shape=(), minval=2.0, maxval=5.0,
        )
        steps_until_next_command = 500

        command = self.sample_command(command_sample_key)

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(12),
            'previous_velocity': jnp.zeros(12),
            'previous_contact': jnp.zeros(4, dtype=bool),
            'command': command,
            'steps_until_next_command': steps_until_next_command,
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'initial_imu_pos': imu_qpos,
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
        rng, cmd_key, sample_key = jax.random.split(state.info['rng'], 3)

        # Physics step:
        motor_targets = self.default_ctrl + action * self.action_scale
        # motor_targets = state.pipeline_state.q[7:] + action * self.action_scale
        pipeline_state = self.pipeline_step(
            state.pipeline_state, motor_targets,
        )

        # Observation data:
        joint_angles = pipeline_state.q[7:]
        joint_velocities = pipeline_state.qd[6:]
        torso_height = jnp.array([
            pipeline_state.site_xpos[self.imu_site_idx][2]
        ])

        # Foot Contact:
        contact = jnp.array([
            collisions.geoms_colliding(pipeline_state, geom_id, self.floor_geom_idx)
            for geom_id in self.feet_geom_idx
        ])

        observation = self.get_observation(
            pipeline_state,
            state.info,
        )

        # Done if joint limits are reached or robot is falling:
        done = self.get_upvector(pipeline_state)[-1] < 0.0

        # Rewards:
        rewards = {
            'tracking_height': (
                self._reward_tracking_height(
                    state.info['command'], torso_height,
                )
            ),
            'tracking_height_error': (
                self._reward_tracking_height_error(
                    state.info['command'], torso_height,
                )
            ),
            'linear_xy_velocity': self._reward_linear_velocity(
                self.get_global_linvel(pipeline_state),
            ),
            'angular_xy_velocity': self._reward_angular_velocity(
                self.get_global_angvel(pipeline_state),
            ),
            'orientation_regularization': self._reward_orientation_regularization(
                self.get_upvector(pipeline_state),
            ),
            'pose_regularization': (
                self._reward_pose_regularization(
                    joint_angles,
                )
            ),
            'torque': self._reward_torques(pipeline_state.actuator_force),
            'action_rate': self._reward_action_rate(action, state.info['previous_action']),
            'acceleration': self._reward_acceleration(
                pipeline_state.qacc,
            ),
            'foot_contact': self._reward_foot_contact(
                pipeline_state, contact,
            ),
            'foot_slip': self._reward_foot_slip(
                pipeline_state, contact,
            ),
            'termination': jnp.float64(
                self._reward_termination(done)
            ) if jax.config.x64_enabled else jnp.float32(
                self._reward_termination(done)
            ),
        }
        rewards = {
            k: v * self.reward_config[k] for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.step_dt, 0.0, 10000.0)

        # State management
        state.info['previous_action'] = action
        state.info['previous_velocity'] = joint_velocities
        state.info['previous_contact'] = contact
        state.info['rewards'] = rewards
        state.info['steps_until_next_command'] -= 1
        state.info['rng'] = rng

        # Command Sampling:
        state.info['command'] = jnp.where(
            state.info['steps_until_next_command'] <= 0,
            self.sample_command(cmd_key),
            state.info['command'],
        )

        # Randomize Command Interval:
        state.info['steps_until_next_command'] = jnp.where(
            done | (state.info['steps_until_next_command'] <= 0),
            500,
            state.info['steps_until_next_command'],
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

        observation = jnp.concatenate([
            noisy_angular_rate,                         # 3
            noisy_projected_gravity,                    # 3
            noisy_joint_positions - self.default_pose,  # 12
            noisy_joint_velocities,                     # 12
            state_info['previous_action'],              # 12
            state_info['command'],                      # 1
        ])
        # Size: 43

        linear_velocity = self.get_local_linvel(pipeline_state)
        global_angular_velocity = self.get_global_angvel(pipeline_state)
        actuator_force = pipeline_state.actuator_force
        torso_height = jnp.array([
            pipeline_state.site_xpos[self.imu_site_idx][2]
        ])

        privileged_observation = jnp.concatenate([
            observation,                                                       # 43
            gyroscope,                                                         # 3
            projected_gravity,                                                 # 3
            linear_velocity,                                                   # 3
            global_angular_velocity,                                           # 3
            q - self.default_pose,                                             # 12
            qd,                                                                # 12
            actuator_force,                                                    # 12
            torso_height,                                                      # 1
        ])
        # Size (Privileged Observations): 92

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

    def _reward_tracking_height_error(
        self, command: jax.Array, global_base_z: jax.Array
    ) -> jax.Array:
        # L1 Error for Tracking of height command (z axis)
        return jnp.sum(jnp.abs(command - global_base_z))

    def _reward_linear_velocity(
        self, global_base_vel: jax.Array,
    ) -> jax.Array:
        # Penalize xy axes base linear velocity
        return jnp.sum(jnp.square(global_base_vel[:2]))

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
        # Penalize large deviations from the default pose
        weight = jnp.array([1.0, 0.0, 0.0] * 4) / 12.0
        return jnp.sum(jnp.square(qpos - self.default_pose) * weight)

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(
        self, action: jax.Array, previous_action: jax.Array
    ) -> jax.Array:
        # Penalize large changes in actions
        return jnp.sum(jnp.square(action - previous_action))

    def _reward_acceleration(
        self, qacc: jax.Array,
    ) -> jax.Array:
        # Penalize Motor/Joint Acceleration
        return jnp.sqrt(jnp.sum(jnp.square(qacc)))

    def _reward_foot_slip(
        self,
        pipeline_state: base.State,
        contact: jax.Array,
    ) -> jax.Array:
        # Penalize foot slip
        foot_velocity = self.get_feet_velocity(pipeline_state)
        foot_velocity_xy = foot_velocity[..., :2]
        velocity_xy_sq = jnp.sum(jnp.square(foot_velocity_xy), axis=-1)
        return jnp.sum(velocity_xy_sq * contact)

    def _reward_foot_contact(
        self,
        pipeline_state: base.State,
        contact: jax.Array,
    ) -> jax.Array:
        # Penalize non contact
        return jnp.sum(~contact)

    def _reward_termination(self, done: jax.Array) -> jax.Array:
        return done

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

    def np_observation(
        self,
        mj_data: mujoco.MjData,
        command: np.ndarray,
        previous_action: np.ndarray,
        add_noise: bool = True,
    ) -> Dict[str, np.ndarray]:
        # Numpy implementation of the observation function:
        def rotate(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
            if len(vec.shape) != 1:
                raise ValueError('vec must have no batch dimensions.')
            s, u = quat[0], quat[1:]
            r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
            r = r + 2 * s * np.cross(u, vec)
            return r

        def quat_inv(q: np.ndarray) -> np.ndarray:
            return q * np.array([1, -1, -1, -1])

        base_w = mj_data.qpos[3:7]
        q = mj_data.qpos[7:]
        qd = mj_data.qvel[6:]

        gyroscope = self.get_gyro(mj_data)

        inverse_trunk_rotation = quat_inv(base_w)
        projected_gravity = rotate(
            jnp.array([0, 0, -1]), inverse_trunk_rotation,
        )

        q = mj_data.qpos[7:]
        qd = mj_data.qvel[6:]

        if add_noise:
            gyroscope = gyroscope + np.random.uniform(
                low=-self.noise_config.gyroscope,
                high=self.noise_config.gyroscope,
                size=gyroscope.shape,
            )
            projected_gravity = projected_gravity + np.random.uniform(
                low=-self.noise_config.gravity_vector,
                high=self.noise_config.gravity_vector,
                size=projected_gravity.shape,
            )
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
            gyroscope,
            projected_gravity,
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
    ) -> Dict[str, np.ndarray]:
        def rotate(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
            if len(vec.shape) != 1:
                raise ValueError('vec must have no batch dimensions.')
            s, u = quat[0], quat[1:]
            r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
            r = r + 2 * s * np.cross(u, vec)
            return r

        def quat_inv(q: np.ndarray) -> np.ndarray:
            return q * np.array([1, -1, -1, -1])
        # Set to Correct Data Type:
        joint_positions = np.asarray(motor_state.q, dtype=np.float32)
        joint_velocities = np.asarray(motor_state.qd, dtype=np.float32)
        gyroscope = np.asarray(imu_state.gyroscope, dtype=np.float32)
        base_rotation = np.asarray(imu_state.quaternion, dtype=np.float32)

        inverse_base_rotation = quat_inv(base_rotation)
        projected_gravity = rotate(
            np.array([0.0, 0.0, -1.0]),
            inverse_base_rotation,
        )

        observation = np.concatenate([
            gyroscope,
            projected_gravity,
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
