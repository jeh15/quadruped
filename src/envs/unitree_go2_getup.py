# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# NOTE: This has been adapted from the mujoco_playground repository (https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/getup.py)
# Original license has been included above.

"""Fall recovery task for the Go2."""

from typing import Any
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
from brax.base import Motion, Transform, System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, html

import mujoco

# Types:
PRNGKey = jax.Array


@flax.struct.dataclass
class RewardConfig:
    # Rewards:
    tracking_linear_velocity: float = 1.5
    tracking_angular_velocity: float = 0.8
    # Penalties / Regularization Terms:
    orientation_regularization: float = -5.0
    linear_z_velocity: float = -2.0
    angular_xy_velocity: float = -0.05
    torque: float = -2e-4
    action_rate: float = -0.01
    stand_still: float = -0.5
    termination: float = -1.0
    foot_slip: float = -0.1
    # Gait Terms:
    air_time: float = 0.2
    target_air_time: float = 0.1
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.25
    kernel_alpha: float = 1.0


@flax.struct.dataclass
class NoiseConfig:
    joint_position: float = 0.03
    joint_velocity: float = 1.5
    gyroscope: float = 0.2
    gravity_vector: float = 0.05


def domain_randomize(sys: System, rng: PRNGKey) -> tuple[System, System]:
    @jax.vmap
    def randomize_parameters(rng):
        key, subkey = jax.random.split(rng)
        # friction
        friction = jax.random.uniform(subkey, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)

        # actuator
        key, subkey = jax.random.split(subkey)
        gain_range = (-5, 5)
        param = jax.random.uniform(
            subkey, (1,), minval=gain_range[0], maxval=gain_range[1]
        ) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)

        return friction, gain, bias

    friction, gain, bias = randomize_parameters(rng)

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'actuator_gainprm': 0,
        'actuator_biasprm': 0,
    })

    sys = sys.tree_replace({
        'geom_friction': friction,
        'actuator_gainprm': gain,
        'actuator_biasprm': bias,
    })  # type: ignore

    return sys, in_axes


class UnitreeGo2Env(PipelineEnv):
    """Environment for training the Unitree Go1 quadruped joystick policy in MJX."""

    def __init__(
        self,
        filename: str = 'unitree_go2/scene_mjx.xml',
        config: RewardConfig = RewardConfig(),
        random_position_prob: float = 0.5,
        action_scale: float = 0.3,
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

        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        self.kernel_sigma = config.kernel_sigma
        self.kernel_alpha = config.kernel_alpha
        self.target_air_time = config.target_air_time
        config_dict = flax.serialization.to_state_dict(config)
        del config_dict['kernel_sigma']
        del config_dict['kernel_alpha']
        del config_dict['target_air_time']
        self.reward_config = config_dict

        self.noise_config = NoiseConfig()

        self.base_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base_link'
        )
        self.random_position_prob = random_position_prob
        self._action_scale = action_scale
        self.init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self.init_qd = jnp.zeros(sys.nv)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos[7:])
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
        self.ctrl_lb = jnp.array([-0.9472, -1.4, -2.6227] * 4)
        self.ctrl_ub = jnp.array([0.9472, 2.5, -0.84776] * 4)
        feet_site = [
            'front_left_foot',
            'front_right_foot',
            'hind_left_foot',
            'hind_right_foot',
        ]
        feet_site_idx = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_idx), 'Site not found.'
        self.feet_site_idx = np.array(feet_site_idx)
        calf_body = [
            'front_left_calf',
            'front_right_calf',
            'hind_left_calf',
            'hind_right_calf',
        ]
        calf_body_idx = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, c)
            for c in calf_body
        ]
        assert not any(id_ == -1 for id_ in calf_body_idx), 'Body not found.'
        imu_site_idx = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'imu')
        assert not any(id_ == -1 for id_ in imu_site_idx), 'IMU site not found.'
        self.calf_body_idx = np.array(calf_body_idx)
        self.foot_radius = 0.022
        self.history_length = 15
        self.num_observations = 31
        self.energy_budget = 1000.0

    def get_random_position(self, rng: PRNGKey) -> jax.Array:
        rng, rotation_key, translation_key = jax.random.split(rng, 3)

        qpos = jnp.zeros(self.sys.nq)

        # Random Body Position and Orientation:
        height = 0.5
        qpos = qpos.at[2].set(height)
        rotation = jax.random.normal(
            rotation_key, shape=(4,),
        )
        rotation /= jnp.linalg.norm(rotation) + 1e-6
        qpos = qpos.at[3:7].set(rotation)

        # Random Joint Angles:
        joint_angles = jax.random.uniform(
            translation_key,
            shape=(12,),
            minval=self.joint_lb,
            maxval=self.joint_ub,
        )
        qpos = qpos.at[7:].set(joint_angles)

        return qpos

    def reset(self, rng: PRNGKey) -> State:
        rng, sample_key, position_key = jax.random.split(rng, 3)

        # Random position:
        qpos = jnp.where(
            jax.random.bernoulli(sample_key, self.random_position_prob),
            self.get_random_position(position_key),
            self.init_q,
        )

        # Random velocity:
        rng, key = jax.random.split(rng)
        qvel = jnp.zeros(self.sys.nv)
        random_body_velocity = jax.random.uniform(
            key, shape=(6,), minval=-0.5, maxval=0.5,
        )
        qvel = qvel.at[:6].set(random_body_velocity)

        pipeline_state = self.pipeline_init(q=qpos, qd=qvel, act=qpos[7:])

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(12),
            'previous_velocity': jnp.zeros(12),
            'previous_contact': jnp.zeros(4, dtype=bool),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'step': 0,
        }

        observation_history = jnp.zeros(
            self.history_length * self.num_observations,
        )
        # Observation Tests:
        observation = self.get_observation(
            pipeline_state, state_info, observation_history,
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
        rng, cmd_rng, kick_noise_key = jax.random.split(state.info['rng'], 3)
        # Physics step:
        # motor_targets = self.default_ctrl + action * self._action_scale
        motor_targets = state.pipeline_state.q[7:] + action * self._action_scale
        motor_targets = jnp.clip(motor_targets, self.ctrl_lb, self.ctrl_ub)
        pipeline_state = self.pipeline_step(
            state.pipeline_state, motor_targets,
        )
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        observation = self.get_observation(
            pipeline_state,
            state.info,
            state.obs,
        )
        joint_angles = pipeline_state.q[7:]
        joint_velocities = pipeline_state.qd[6:]
        joint_torques = pipeline_state.actuator_force

        # Done if robot exceeds energy budget
        energy = jnp.sum(
            jnp.abs(pipeline_state.actuator_force) * joint_velocities
        )
        done = energy > self.energy_budget

        # Rewards:
        body_height = pipeline_state.site_xpos[self.imu_site_idx][2]

        rewards = {
            'orientation': (
                self._reward_orientation()
            ),
            'linear_z_velocity': self._reward_vertical_velocity(xd),
            'angular_xy_velocity': self._reward_angular_velocity(xd),
            'orientation_regularization': self._reward_orientation_regularization(x),
            'torque': self._reward_torques(pipeline_state.qfrc_actuator[6:]),
            'action_rate': self._reward_action_rate(action, state.info['previous_action']),
            'stand_still': self._reward_stand_still(
                state.info['command'], joint_angles,
            ),
            'foot_slip': self._reward_foot_slip(
                pipeline_state, contact_filt_cm,
            ),
            'air_time': self._reward_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
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

        # state management
        state.info['previous_action'] = action
        state.info['previous_velocity'] = joint_velocities
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # Proxy Metrics:
        state.metrics['total_distance'] = math.normalize(
            x.pos[self.base_idx - 1])[1]
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
        observation_history: jax.Array,
    ) -> jax.Array:
        """
            Observation: [
                gyroscope,
                gravity_vector,
                motor_positions,
                motor_velocities,
                previous_action,
            ]
        """
        inverse_trunk_rotation = math.quat_inv(pipeline_state.x.rot[0])
        body_frame_angular_rate = math.rotate(
            pipeline_state.xd.ang[0], inverse_trunk_rotation,
        )
        projected_gravity = math.rotate(
            jnp.array([0, 0, -1]), inverse_trunk_rotation,
        )

        q = pipeline_state.q[7:]
        qd = pipeline_state.qd[6:]

        # Gyroscope Noise:
        state_info['rng'], noise_key = jax.random.split(state_info['rng'])
        gyroscope_noise = jax.random.uniform(
            noise_key,
            shape=body_frame_angular_rate.shape,
            minval=-self.noise_config.gyroscope, 
            maxval=self.noise_config.gyroscope,
        )
        noisy_angular_rate = body_frame_angular_rate + gyroscope_noise

        # Gravity noise:
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
            noisy_angular_rate,
            noisy_projected_gravity,
            noisy_joint_positions - self.default_pose,
            noisy_joint_velocities,
            state_info['previous_action'],
        ])

        # stack observations through time
        observation = jnp.roll(
            observation_history, observation.size
        ).at[:observation.size].set(observation)

        return observation

    def _reward_vertical_velocity(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2])

    def _reward_angular_velocity(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(xd.ang[0, :2]))

    def _reward_orientation_regularization(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jnp.array([0.0, 0.0, 1.0])
        deviation = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(deviation[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(
        self, action: jax.Array, previous_action: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(action - previous_action))

    def _reward_tracking_velocity(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        base_velocity = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        error = jnp.sum(jnp.square(commands[:2] - base_velocity[:2]))
        return jnp.exp(-error / self.kernel_sigma)

    def _reward_tracking_yaw_rate(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_yaw_rate = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        error = jnp.square(commands[2] - base_yaw_rate[2])
        return jnp.exp(-error / self.kernel_sigma)

    def _reward_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Flight Phase Reward:
        reward_air_time = jnp.sum((air_time - self.target_air_time) * first_contact)
        reward_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return reward_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jnp.sum(jnp.abs(joint_angles - self.default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, pipeline_state: base.State, contact_filter: jax.Array
    ) -> jax.Array:
        # Foot Velocity:
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self.feet_site_idx]
        feet_offset = pos - pipeline_state.xpos[self.calf_body_idx]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self.calf_body_idx - 1
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(foot_vel[:, :2]) * contact_filter.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

    def np_observation(
        self,
        mj_data: mujoco.MjData,
        command: np.ndarray,
        previous_action: np.ndarray,
        observation_history: np.ndarray,
    ) -> np.ndarray:
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
        base_dw = mj_data.qvel[3:6]
        q = mj_data.qpos[7:]

        inverse_trunk_rotation = quat_inv(base_w)
        body_frame_yaw_rate = rotate(
            base_dw, inverse_trunk_rotation,
        )[2]
        projected_gravity = rotate(
            jnp.array([0, 0, -1]), inverse_trunk_rotation,
        )

        new_observation = np.concatenate([
            np.array([body_frame_yaw_rate]),
            projected_gravity,
            command,
            q - self.default_ctrl,
            previous_action,
        ])

        # clip, noise
        new_observation = (
            jnp.clip(new_observation, -100.0, 100.0)
            + self._obs_noise * np.random.uniform(
                low=-1, high=1, size=new_observation.shape,
            )
        )
        # stack observations through time
        observation = np.roll(observation_history, new_observation.size)
        observation[:new_observation.size] = new_observation

        return observation


envs.register_environment('unitree_go2', UnitreeGo2Env)


def main(argv=None):
    env = UnitreeGo2Env(filename='unitree_go2/scene_barkour_hfield_mjx.xml')
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