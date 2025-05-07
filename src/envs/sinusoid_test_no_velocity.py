"""
    Unitree Go2 Environment:
        Simple Feet Position Test
"""

from typing import Any, Dict
from absl import app
import os

import flax.serialization
import jax
import jax.numpy as jnp

import numpy as np
from scipy.spatial.transform import Rotation as R

import flax.struct
import flax.serialization

from brax import base
from brax import envs
from brax import math
from brax.base import System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html

import mujoco


# Types:
PRNGKey = jax.Array

@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_pose: float = 1.0
    # Pose Regularizations:
    pose_regularization: float = -5.0
    abduction_regularization: float = -1.0
    # Energy Regularization Terms:
    torque: float = -2e-4
    action_rate: float = -0.01
    mechanical_power: float = -1e-3
    acceleration: float = -1e-3
    # Auxilary Terms:
    termination: float = -1.0
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.05
    kernel_alpha: float = 1.0


@flax.struct.dataclass
class NoiseConfig:
    joint_position: float = 0.05


def domain_randomize(sys: System, rng: PRNGKey) -> tuple[System, System]:
    @jax.vmap
    def randomize_parameters(rng):
        # Joint Friction:
        rng, key = jax.random.split(rng)
        frictionloss = sys.dof_frictionloss * jax.random.uniform(
            key, shape=(12,), minval=0.9, maxval=1.1,
        )
        dof_frictionloss = sys.dof_frictionloss.at[:].set(frictionloss)

        # Armature:
        rng, key = jax.random.split(rng)
        armature = sys.dof_armature * jax.random.uniform(
            key, shape=(12,), minval=1.0, maxval=1.05,
        )
        dof_armature = sys.dof_armature.at[:].set(armature)

        # Link mass randomization:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, (sys.nbody,), minval=0.9, maxval=1.1,
        )
        body_mass = sys.body_mass.at[:].set(sys.body_mass * delta)

        return (
            dof_frictionloss,
            dof_armature,
            body_mass,
        )

    (
        dof_frictionloss,
        dof_armature,
        body_mass,
    ) = randomize_parameters(rng)

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'dof_frictionloss': 0,
        'dof_armature': 0,
        'body_mass': 0,
    })

    sys = sys.tree_replace({
        'dof_frictionloss': dof_frictionloss,
        'dof_armature': dof_armature,
        'body_mass': body_mass,
    })  # type: ignore

    return sys, in_axes


class UnitreeGo2Env(PipelineEnv):
    """Environment for training the Unitree Go2 quadruped joystick policy in MJX."""

    def __init__(
        self,
        filename: str = 'unitree_go2/scene_mjx_fixed.xml',
        config: RewardConfig = RewardConfig(),
        action_scale: float = 0.3,
        foot_height_amplitude: float = 0.1,
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

        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        self.kernel_sigma = config.kernel_sigma
        self.kernel_alpha = config.kernel_alpha
        config_dict = flax.serialization.to_state_dict(config)
        del config_dict['kernel_sigma']
        del config_dict['kernel_alpha']
        self.reward_config = config_dict

        self.noise_config = NoiseConfig()

        self.floor_geom_idx = self.sys.mj_model.geom('floor').id
        self.base_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
        )
        self.base_link_mass = self.sys.mj_model.body_subtreemass[self.base_idx]

        self.action_scale = action_scale
        self.foot_height_amplitude = foot_height_amplitude
        self.init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self.init_qd = jnp.zeros(sys.nv)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos)
        self.default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
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
        self.ctrl_lb = jnp.array([
            -0.9472, -1.4, -2.6227,
            -0.9472, -1.4, -2.6227,
            -0.9472, -0.4236, -2.6227,
            -0.9472, -0.4236, -2.6227,
        ])
        self.ctrl_ub = jnp.array([
            0.9472, 2.5, -0.84776,
            0.9472, 2.5, -0.84776,
            0.9472, 2.5, -0.84776,
            0.9472, 2.5, -0.84776,
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

        # Get Feet Positions:
        self.default_feet_position = np.array([
            [0.217, -0.142, -0.308],
            [0.217, 0.142, -0.308],
            [-0.169, -0.142, -0.308],
            [-0.169, 0.142, -0.308],
        ])

        # Constants:
        self.num_observations = 25
        self.num_privileged_observations = 85

    def sample_command(self, rng: jax.Array) -> jax.Array:
        command_range = [-self.foot_height_amplitude, self.foot_height_amplitude]
        key, subkey = jax.random.split(rng)
        new_cmd = jax.random.uniform(
            subkey,
            (1,),
            minval=command_range[0],
            maxval=command_range[1],
        )
        return new_cmd
    
    def reset(self, rng: PRNGKey) -> State:  # pytype: disable=signature-mismatch
        # Initial Position:
        rng, key = jax.random.split(rng)
        delta = jax.random.uniform(
            key, shape=(12,), minval=-0.1, maxval=0.1,
        )
        qpos = self.init_q.at[:].set(self.init_q + delta)

        # Initial Velocity:
        rng, key = jax.random.split(rng)
        qvel = self.init_qd.at[:].set(
            jax.random.uniform(key, (12,), minval=-0.5, maxval=0.5)
        )

        pipeline_state = self.pipeline_init(qpos, qvel)

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(12),
            'previous_velocity': jnp.zeros(12),
            'command': self.sample_command(key),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
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
        joint_angles = pipeline_state.q
        joint_velocities = pipeline_state.qd
        feet_position = self.get_feet_pos(pipeline_state)

        # Done if joint limits are reached:
        done = jnp.any(joint_angles < self.joint_lb)
        done |= jnp.any(joint_angles > self.joint_ub)

        # Rewards:
        rewards = {
            'tracking_pose': (
                self._reward_tracking_pose(state.info['command'], feet_position)
            ),
            'pose_regularization': (
                self._reward_pose_regularization(feet_position)
            ),
            'abduction_regularization': (
                self._reward_abduction_regularization(joint_angles)
            ),
            'torque': self._reward_torques(pipeline_state.actuator_force),
            'action_rate': self._reward_action_rate(action, state.info['previous_action']),
            'mechanical_power': self._reward_mechanical_power(
                joint_velocities, pipeline_state.actuator_force,
            ),
            'acceleration': self._reward_acceleration(
                pipeline_state.qacc,
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
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # Sample new command if more than 100 timesteps achieved (~2.0 Seconds)
        state.info['command'] = jnp.where(
            state.info['step'] > 100,
            self.sample_command(cmd_rng),
            state.info['command'],
        )
        # Reset the step counter when done
        state.info['step'] = jnp.where(
            done | (state.info['step'] > 100), 0, state.info['step']
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
                previous_action,
                command,
            ]
        """
        q = pipeline_state.q
        qd = pipeline_state.qd

        # Joint position noise:
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        joint_position_noise = jax.random.uniform(
            noise_key,
            shape=q.shape,
            minval=-self.noise_config.joint_position,
            maxval=self.noise_config.joint_position,
        )
        noisy_joint_positions = q + joint_position_noise


        observation = jnp.concatenate([
            noisy_joint_positions - self.default_pose,  # 12
            state_info['previous_action'],              # 12
            state_info['command'],                      # 1
        ])
        # Size: 25


        actuator_force = pipeline_state.actuator_force
        feet_position = self.get_feet_pos(pipeline_state).ravel()
        feet_velocity = self.get_feet_velocity(pipeline_state).ravel()

        privileged_observation = jnp.concatenate([
            observation,               # 25
            q - self.default_pose,     # 12
            qd,                        # 12
            actuator_force,            # 12
            feet_position,             # 12
            feet_velocity,             # 12
        ])
        # Size: 85

        return {
            'state': observation,
            'privileged_state': privileged_observation,
        }

    def _reward_tracking_pose(
        self, command: jax.Array, foot_position: jax.Array,
    ) -> jax.Array:
        # Tracking of foot positions:
        desired_foot_height = self.default_feet_position[:, -1] + command
        error = jnp.sum(jnp.square(desired_foot_height - foot_position[:, -1]))
        return jnp.exp(-error / self.kernel_sigma)
    
    def _reward_pose_regularization(
        self, foot_position: jax.Array,
    ) -> jax.Array:
        # Penalize foot XY deviation from default position:
        return jnp.sum(jnp.square(self.default_feet_position[:, :-1] - foot_position[:, :-1]))

    def _reward_abduction_regularization(
        self, qpos: jax.Array,
    ) -> jax.Array:
        # Penalize abduction deviation:
        abduction_joints = jnp.reshape(qpos, shape=(4, 3))[:, 0]
        return jnp.sum(jnp.square(jnp.zeros(4) - abduction_joints))

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

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

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
    ) -> np.ndarray:
        q = mj_data.qpos

        if add_noise:
            q = q + np.random.uniform(
                low=-self.noise_config.joint_position,
                high=self.noise_config.joint_position,
                size=q.shape,
            )

        observation = np.concatenate([
            q - self.default_ctrl,
            previous_action,
            command,
        ])

        return {
            'state': observation,
            'privileged_state': np.zeros((self.num_privileged_observations,)),
        }
    
    def hardware_observation(
        self,
        motor_state: Any,
        command: np.ndarray,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        # Set to Correct Data Type:
        joint_positions = np.asarray(motor_state.q, dtype=np.float32)

        # Cast to float64:
        joint_positions = joint_positions.astype(np.float64)

        observation = np.concatenate([
            joint_positions - self.default_ctrl,
            previous_action,
            command,
        ])

        return {
            'state': observation,
            'privileged_state': np.zeros((self.num_privileged_observations,)),
        }

envs.register_environment('unitree_go2', UnitreeGo2Env)


def main(argv=None):
    env = UnitreeGo2Env(filename='unitree_go2/scene_mjx_fixed.xml')
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
