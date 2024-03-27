import dataclasses
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class MLP(nn.Module):
    """MLP module adopted from Brax."""
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(
                hidden_size,
                name=f'hidden_{i}',
                kernel_init=self.kernel_init,
                use_bias=self.bias)(
                    hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden