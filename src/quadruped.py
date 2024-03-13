from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
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
        filepath: str,
        backend: str = 'generalized',
        params: config_dict.ConfigDict = config,
        **kwargs,
    ):
        sys = mjcf.load(filepath)

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

        # Set Configuration:
        self._forward_reward_weight = params.forward_reward_weight
        self._ctrl_cost_weight = params.ctrl_cost_weight
        self._healthy_reward = params.healthy_reward
        self._terminate_when_unhealthy = params.terminate_when_unhealthy
        self._healthy_z_range = params.healthy_z_range
        self._reset_noise_scale = params.reset_noise_scale
        self._exclude_current_positions_from_observation = (
            params.exclude_current_positions_from_observation
        )

    def reset(self, rng: jnp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        q = self.initial_q
        qd = jnp.zeros((self.qd_size,))
        pipeline_state = self.pipeline_init(q, qd)

        obs = self._get_states(pipeline_state)

        # Need to implement a way to get feet positions in the world.
        # Reward based on foot contact with the ground.
        # Reward based on foot height.
        # Reward based on duty cycle of feet.

        # Reward Function:
        reward = jnp.array([0.0])
        done = jnp.array([0])
        zero = jnp.array([0.0])
        metrics = {
            'reward_forward': zero,
            'reward_linear_velocity': zero,
            'reward_ctrl': zero,
            'reward_foot_height': zero,
            'reward_duty_cycle': zero,
            'reward_termination': zero,
            'position': jnp.zeros(3),
            'orientation': jnp.zeros(3),
            'linear_velocity': jnp.zeros(3),
            'angular_velocity': jnp.zeros(3),
        }

        return State(pipeline_state, obs, reward, done, metrics)

    def step(
        self,
        state: State,
        action: jax.typing.ArrayLike,
    ) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_states(pipeline_state)
        # Reward Function:
        reward = 0
        done = 0
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done,
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
