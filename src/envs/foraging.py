from typing import Tuple, Dict, Any
import os

import jax
import jax.numpy as jnp

import mujoco

from brax.io import mjcf
from brax.envs.base import PipelineEnv, State


# Environment:
class Foraging(PipelineEnv):
    """ Environment for training Cart Pole balancing """

    def __init__(
        self,
        filename: str = 'double_integrator/foraging_scene.xml',
        backend: str = 'mjx',
        **kwargs,
    ):
        # Initialize System:
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
        n_frames = kwargs.pop('n_frames', int(self.step_dt / sys.opt.timestep))
        super().__init__(sys, backend=backend, n_frames=n_frames)

        # Body Index:
        self.body_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "body",
        )
        self.mass = self.sys.body_mass[self.body_idx]

        # Constants: (TODO: Move to config and domain randomization):
        self.reward_scale = 0.001
        self.energy_cap = 10.0
        self.metabolic_rate = -0.02
        self.work_scale = 0.0
        self.kinetic_energy_scale = 0.0
        self.foraging_scale = 1.0
        self.foraging_rate = 1.0
        self.food_patch_x = 2.0
        self.food_patch_y = 2.0
        self.food_patch_r = 1.0

    def reset(self, rng: jax.Array) -> State:
        key, q_key, qd_key, food_patch_key = jax.random.split(rng, 4)

        # q structure: [x y]
        q_init = jax.random.uniform(q_key, (2,), minval=-4.0, maxval=4.0)

        # qd structure: [dx dy]
        qd_init = jax.random.uniform(qd_key, (2,), minval=-0.0, maxval=0.0)

        # Food Patch:
        # food_patch_q = jax.random.uniform(food_patch_key, (2,), minval=-4.0, maxval=4.0)
        food_patch_q = jnp.array([self.food_patch_x, self.food_patch_y])

        # Initialize State:
        pipeline_state = self.pipeline_init(q_init, qd_init)

        # Initialize Rewards:
        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        # State Info and Metrics:
        state_info = {
            'energy_state': jnp.array([self.energy_cap]),
            'work': 0.0,
            'kinetic_energy': 0.0,
            'previous_state': {
                'q': pipeline_state.q,
                'qd': pipeline_state.qd,
            },
            'food_patch': food_patch_q,
        }

        observation = self.get_observation(pipeline_state, state_info)

        metrics = {
            'energy_state': jnp.array([self.energy_cap]),
            'work': 0.0,
            'kinetic_energy': 0.0,
            'metabolic_rate': self.metabolic_rate,
            'foraging_reward': 0.0,
        }

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
            info=state_info,
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        # Forward Physics Step:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # Get Observation from new state:
        observation = self.get_observation(pipeline_state, state.info)

        # Extract States:
        x, y = pipeline_state.q

        # Foraging Patch:
        # foraging = (
        #     ((x - state.info['food_patch'][0]) ** 2 + (y - state.info['food_patch'][1]) ** 2) < (self.food_patch_r ** 2)
        # )
        # foraged_energy = foraging * self.foraging_rate
        # energy_state += foraged_energy

        energy_rewards = {
            'energy_state': state.info['energy_state'],
            'work': self.work_scale * self._work_reward(pipeline_state, state.info),
            'kinetic_energy': (
                self.kinetic_energy_scale * self._kinetic_energy_reward(pipeline_state)
            ),
            'metabolic_rate': self.metabolic_rate,
            'foraging_reward': (
                self.foraging_scale * self._foraging_reward(pipeline_state, state.info)
            ),
        }

        # Reward:
        energy_state = jnp.clip(
            sum(energy_rewards.values()), -jnp.inf, self.energy_cap,
        )

        # Unbound Reward:
        reward = self.reward_scale * energy_state[0]

        # Bounded Reward:
        # normalized_reward = (energy_state - self.energy_cap) ** 2
        # unscaled_reward = jnp.exp(-normalized_reward / 0.25)
        # reward = self.reward_scale * unscaled_reward[0]

        # Terminate if outside the range:
        outside_x = jnp.abs(x) > 5.0
        outside_y = jnp.abs(y) > 5.0
        depleted_energy = energy_state[0] < 0.0
        done = outside_x | outside_y | depleted_energy
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        # Update State Info:
        state.info['energy_state'] = energy_state
        state.info['work'] = energy_rewards['work']
        state.info['kinetic_energy'] = energy_rewards['kinetic_energy']
        state.info['previous_state']['q'] = pipeline_state.q
        state.info['previous_state']['qd'] = pipeline_state.qd
        state.info['food_patch'] = state.info['food_patch']

        # Track Metrics:
        # state.metrics['energy_state'] = state.info['energy_state']
        # state.metrics['work'] = state.info['work']
        # state.metrics['kinetic_energy'] = state.info['kinetic_energy']
        state.metrics.update(energy_rewards)

        # Update State object:
        state = state.replace(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
        )
        return state

    def _energy_rewards(
        self, pipeline_state: State, state_info: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q, qd = pipeline_state.q, pipeline_state.qd

        # Acceleration and Displacement:
        acceleration = jnp.linalg.norm(
            (qd - state_info['previous_state']['qd']) / self.step_dt,
        )
        velocity = jnp.linalg.norm(qd)
        displacement = jnp.linalg.norm(q - state_info['previous_state']['q'])

        # Energy Calculations:
        kinetic_energy = 0.5 * jnp.dot(velocity, velocity)
        work = self.mass * acceleration * displacement

        return work, kinetic_energy

    def _work_reward(
        self, pipeline_state: State, state_info: Dict[str, Any],
    ) -> jnp.ndarray:
        q, qd = pipeline_state.q, pipeline_state.qd

        # Acceleration and Displacement:
        acceleration = jnp.linalg.norm(
            (qd - state_info['previous_state']['qd']) / self.step_dt,
        )
        displacement = jnp.linalg.norm(q - state_info['previous_state']['q'])

        return self.mass * acceleration * displacement

    def _kinetic_energy_reward(
        self, pipeline_state: State,
    ) -> jnp.ndarray:
        qd = pipeline_state.qd

        # Magnitude of Velocity:
        velocity = jnp.linalg.norm(qd)

        return 0.5 * jnp.dot(velocity, velocity)

    def _foraging_reward(
        self, pipeline_state: State, state_info: Dict[str, Any],
    ) -> jnp.ndarray:
        x, y = pipeline_state.q

        # Continuous Reward:
        foraging_reward = jnp.exp(
            -1 * (
                (x - state_info['food_patch'][0]) ** 2
                + (y - state_info['food_patch'][1]) ** 2
            )
        )

        return foraging_reward

    def get_observation(
        self, pipeline_state: State, state_info: Dict[str, Any],
    ) -> jnp.ndarray:
        # Observation: [x_displacement, y_displacement, dx, dy, previous_energy]
        x, y = pipeline_state.q
        dx, dy = pipeline_state.qd

        # Distance to food patch:
        x_displacement = x - state_info['food_patch'][0]
        y_displacement = y - state_info['food_patch'][1]

        return jnp.array([
            x_displacement, y_displacement, dx, dy, state_info['energy_state'][0],
        ])
