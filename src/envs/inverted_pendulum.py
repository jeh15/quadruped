import os

import jax
import jax.numpy as jnp
import numpy as np

from brax import envs
from brax.io import mjcf
from brax.envs.base import PipelineEnv, State
import mujoco

# Environment:
class InvertedPendulum(PipelineEnv):
    """ Environment for training Cart Pole balancing """

    def __init__(self, filename: str = 'cartpole/inverted_pendulum.xml', backend: str = 'mjx', **kwargs):
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

    def reset(self, rng: jax.Array) -> State:
        key, theta_key, qd_key = jax.random.split(rng, 3)

        theta_init = jax.random.uniform(theta_key, (1,), minval=-0.1, maxval=0.1)[0]
        
        # q structure: [x th]
        q_init = jnp.array([0.0, theta_init])
        
        # qd structure: [dx dth]
        qd_init = jax.random.uniform(qd_key, (2,), minval=-0.1, maxval=0.1)        
        
        # Initialize State:
        pipeline_state = self.pipeline_init(q_init, qd_init)

        # Initialize Rewards:
        reward, done = jnp.zeros(2)
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        # Get observation for RL Algorithm (Input to our neural net):
        observation = self.get_observation(pipeline_state)

        # Metrics:
        metrics = {
            'rewards': reward,
            'observation': observation,
        }
        metrics = {}

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        # Forward Physics Step:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        # Get Observation from new state:
        observation = self.get_observation(pipeline_state)

        # Extract States:
        x, th = pipeline_state.q

        # Terminate if outside the range of the rail and pendulum is past the rail:
        outside_x = jnp.abs(x) > 1.0
        outside_th = jnp.abs(th) > jnp.pi / 2.0
        done = outside_x | outside_th
        done = jnp.float64(done) if jax.config.x64_enabled else jnp.float32(done)

        # Calculate Reward:
        reward = jnp.cos(th)

        # Update State object:
        state = state.replace(
            pipeline_state=pipeline_state, obs=observation, reward=reward, done=done,
        )
        return state

    def get_observation(self, pipeline_state: State) -> jnp.ndarray:
        # Observation: [x, th, dx, dth]
        return jnp.concatenate([pipeline_state.q, pipeline_state.qd])

    def mujoco_get_observation(
        self,
        mj_data: mujoco.MjData,
    ) -> np.ndarray:
        # Numpy implementation of the observation function:
        return np.concatenate([mj_data.qpos, mj_data.qvel])


envs.register_environment('inverted_pendulum', InvertedPendulum)
