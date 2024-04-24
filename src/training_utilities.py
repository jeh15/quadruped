from typing import Callable, Sequence, Tuple

import jax

from brax import envs
from src.module_types import Transition
import src.module_types as types


State = envs.State
Env = envs.Env


def policy_step(
    env: Env,
    state: State,
    policy: types.Policy,
    key: types.PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    actions, policy_data = policy(state.obs, key)
    next_state = env.step(state, actions)
    state_data = {x: next_state.info[x] for x in extra_fields}
    return next_state, Transition(
        observation=state.obs,
        action=actions,
        reward=next_state.reward,
        termination=next_state.done,
        next_observation=next_state.obs,
        extras={
            'policy_data': policy_data,
            'state_data': state_data,
        },
    )


def unroll_policy_steps(
    env: Env,
    state: State,
    policy: types.Policy,
    key: types.PRNGKey,
    num_steps: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    @jax.jit
    def f(carry, unused_t):
        state, key = carry
        key, subkey = jax.random.split(key)
        state, transition = policy_step(
            env,
            state,
            policy,
            key,
            extra_fields,
        )
        return (state, subkey), transition

    (final_state, _), transitions = jax.lax.scan(
        f,
        (state, key),
        xs=None,
        length=num_steps,
    )
    return final_state, transitions
