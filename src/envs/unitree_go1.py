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
    feet_air_time: float = 0.2
    # Penalties / Regularization Terms:
    linear_z_velocity: float = -2.0
    angular_xy_velocity: float = -0.05
    orientation: float = -5.0
    torque: float = -2e-4
    action_rate: float = -0.01
    stand_still: float = -0.5
    termination: float = -1.0
    foot_slip: float = -0.1
    # Hyperparameter for exponential kernel:
    kernel_sigma: float = 0.25
    # Target air time for feet:
    target_air_time: float = 0.1


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


class UnitreeGo1Env(PipelineEnv):
    """Environment for training the Unitree Go1 quadruped joystick policy in MJX."""

    def __init__(
        self,
        filename: str = 'unitree_go1/scene.xml',
        config: RewardConfig = RewardConfig(),
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        train_fast_cmd: bool = False,
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
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        self.kernel_sigma = config.kernel_sigma
        self.target_air_time = config.target_air_time
        config_dict = flax.serialization.to_state_dict(config)
        del config_dict['kernel_sigma']
        del config_dict['target_air_time']
        self.reward_config = config_dict

        self.trunk_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'trunk'
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._train_fast_cmd = train_fast_cmd
        self.init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self.init_qd = jnp.zeros(sys.nv)
        self.default_pose = jnp.array(sys.mj_model.keyframe('home').qpos[7:])
        self.default_ctrl = jnp.array(sys.mj_model.keyframe('home').ctrl)
        self.ctrl_lb = jnp.array([-0.863, -0.686, -2.818] * 4)
        self.ctrl_ub = jnp.array([0.863, 4.501, -0.888] * 4)
        feet_site = [
            'foot_front_left',
            'foot_hind_left',
            'foot_front_right',
            'foot_hind_right',
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
        self.feet_site_id = np.array(feet_site_id)
        calf_body = [
            'front_left_calf',
            'hind_left_calf',
            'front_right_calf',
            'hind_right_calf',
        ]
        calf_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, c)
            for c in calf_body
        ]
        assert not any(id_ == -1 for id_ in calf_body_id), 'Body not found.'
        self.calf_body_id = np.array(calf_body_id)
        self._foot_radius = 0.023
        self.history_length = 15
        self.num_observations = 31

    def sample_command(self, rng: PRNGKey) -> jax.Array:
        if self._train_fast_cmd:
            lin_vel_x = [0.75, 3.0]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
        else:
            lin_vel_x = [-0.6, 1.5]  # min max [m/s]
            lin_vel_y = [-0.8, 0.8]  # min max [m/s]
            ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: PRNGKey) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self.init_q, self.init_qd)

        state_info = {
            'rng': rng,
            'previous_action': jnp.zeros(12),
            'previous_velocity': jnp.zeros(12),
            'command': self.sample_command(key),
            'last_contact': jnp.zeros(4, dtype=bool),
            'feet_air_time': jnp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.keys()},
            'kick': jnp.array([0.0, 0.0]),
            'step': 0,
        }

        observation_history = jnp.zeros(
            self.history_length * self.num_observations,
        )
        observation = self.get_observation(
            pipeline_state, state_info, observation_history,
        )

        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info,
        )  # pytype: disable=wrong-arg-types
        return state

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)

        # kick
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        # physics step
        motor_targets = self.default_pose + action * self._action_scale
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
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        # pytype: disable=attribute-error
        foot_pos = pipeline_state.site_xpos[self.feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        state.info['feet_air_time'] += self.step_dt

        # done if joint limits are reached or robot is falling
        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[self.trunk_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.ctrl_lb)
        done |= jnp.any(joint_angles > self.ctrl_ub)
        done |= pipeline_state.x.pos[self.trunk_idx - 1, 2] < 0.18

        # reward
        rewards = {
            'tracking_linear_velocity': (
                self._reward_tracking_lin_vel(state.info['command'], x, xd)
            ),
            'tracking_angular_velocity': (
                self._reward_tracking_ang_vel(state.info['command'], x, xd)
            ),
            'linear_z_velocity': self._reward_lin_vel_z(xd),
            'angular_xy_velocity': self._reward_ang_vel_xy(xd),
            'orientation': self._reward_orientation(x),
            # pytype: disable=attribute-error
            'torque': self._reward_torques(pipeline_state.qfrc_actuator),
            'action_rate': self._reward_action_rate(action, state.info['previous_action']),
            'stand_still': self._reward_stand_still(
                state.info['command'], joint_angles,
            ),
            'feet_air_time': self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
            'termination': jnp.float64(
                self._reward_termination(done, state.info['step'])
            ) if jax.config.x64_enabled else jnp.float32(
                self._reward_termination(done, state.info['step'])
            ),
        }
        rewards = {
            k: v * self.reward_config[k] for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.step_dt, 0.0, 10000.0)

        # state management
        state.info['kick'] = kick
        state.info['previous_action'] = action
        state.info['previous_velocity'] = joint_vel
        state.info['feet_air_time'] *= ~contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # sample new command if more than 500 timesteps achieved
        state.info['command'] = jnp.where(
            state.info['step'] > 500,
            self.sample_command(cmd_rng),
            state.info['command'],
        )
        # reset the step counter when done
        state.info['step'] = jnp.where(
            done | (state.info['step'] > 500), 0, state.info['step']
        )

        # log total displacement as a proxy metric
        state.metrics['total_dist'] = math.normalize(
            x.pos[self.trunk_idx - 1])[1]
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
        # Observation: [yaw_rate, projected_gravity, command, relative_motor_positions, previous_action]
        inverse_trunk_rotation = math.quat_inv(pipeline_state.x.rot[0])
        body_frame_yaw_rate = math.rotate(
            pipeline_state.xd.ang[0], inverse_trunk_rotation,
        )[2]
        projected_gravity = math.rotate(jnp.array([0, 0, -1]), inverse_trunk_rotation)

        observation = jnp.concatenate([
            jnp.array([body_frame_yaw_rate]) * 0.25,
            projected_gravity,
            state_info['command'] * jnp.array([2.0, 2.0, 0.25]),
            pipeline_state.q[7:] - self.default_pose,
            state_info['previous_action'],
        ])

        # clip, noise
        observation = (
            jnp.clip(observation, -100.0, 100.0)
            + self._obs_noise
            * jax.random.uniform(
                state_info['rng'],
                observation.shape,
                minval=-1,
                maxval=1,
            )
        )
        # stack observations through time
        observation = jnp.roll(
            observation_history, observation.size
        ).at[:observation.size].set(observation)

        return observation

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(
        self, action: jax.Array, previous_action: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(action - previous_action))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(jnp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jnp.exp(
            -lin_vel_error / self.kernel_sigma
        )
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.square(commands[2] - base_ang_vel[2])
        return jnp.exp(-ang_vel_error / self.kernel_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jnp.sum((air_time - self.target_air_time) * first_contact)
        rew_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

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
        self, pipeline_state: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self.feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self.calf_body_id]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self.calf_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)


envs.register_environment('unitree_go1', UnitreeGo1Env)


def main(argv=None):
    env = UnitreeGo1Env()
    rng = jax.random.PRNGKey(0)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    state = reset_fn(rng)

    num_steps = 200
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
