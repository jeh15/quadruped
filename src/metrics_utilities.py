from typing import Callable
import time

import jax
import numpy as np

from brax import envs

import src.module_types as types
from src.training_utilities import unroll_policy_steps


class Evaluator:

    def __init__(
        self,
        env: envs.Env,
        policy_generator: Callable[[types.PolicyParams], types.Policy],
        num_envs: int,
        episode_length: int,
        action_repeat: int,
        key: types.PRNGKey,
    ):
        self.key = key
        self.walltime = 0.0

        env = envs.training.EvalWrapper(env)

        def _evaluation_loop(
            policy_params: types.PolicyParams,
            key: types.PRNGKey,
        ) -> types.State:
            reset_keys = jax.random.split(key, num_envs)
            initial_state = env.reset(reset_keys)
            final_state, _ = unroll_policy_steps(
                env,
                initial_state,
                policy_generator(policy_params),
                key,
                num_steps=episode_length // action_repeat,
            )
            return final_state

        self.evaluation_loop = jax.jit(_evaluation_loop)
        self.steps_per_epoch = episode_length * num_envs

    def evaluate(
        self,
        policy_params: types.PolicyParams,
        training_metrics: types.Metrics,
        aggregate_episodes: bool = True,
    ) -> types.Metrics:
        self.key, subkey = jax.random.split(self.key)

        start_time = time.time()
        state = self.evaluation_loop(policy_params, subkey)
        evaluation_metrics = state.info['eval_metrics']
        evaluation_metrics.active_episodes.block_until_ready()
        epoch_time = time.time() - start_time
        metrics = {}
        for func in [np.mean, np.std]:
            suffix = '_std' if func == np.std else ''
            metrics.update({
                f'eval/episode_{name}{suffix}': (
                    func(value) if aggregate_episodes else value
                )
                for name, value in evaluation_metrics.episode_metrics.items()
            })

        metrics['eval/avg_episode_length'] = np.mean(
            evaluation_metrics.episode_steps,
        )
        metrics['eval/epoch_time'] = epoch_time
        metrics['eval/steps_per_second'] = self.steps_per_epoch / epoch_time
        self.walltime = self.walltime + epoch_time
        metrics = {
            'eval/walltime': self.walltime,
            **training_metrics,
            **metrics,
        }

        return metrics
