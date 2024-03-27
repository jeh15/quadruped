import jax
import jax.numpy as jnp
from flax import linen as nn


class ActorCriticNetwork(nn.Module):
    action_space: int

    def setup(self):
        dtype = jnp.float64
        policy_features = 256
        value_features = 256
        initializer = jax.nn.initializers.lecun_uniform()
        policy_layer_initializer = jax.nn.initializers.variance_scaling(
            scale=0.001,
            mode='fan_in',
            distribution='uniform',
        )
        value_layer_initializer = jax.nn.initializers.lecun_uniform()
        self.dense_1 = nn.Dense(
            features=policy_features,
            name='dense_1',
            kernel_init=initializer,
            dtype=dtype,
        )
        self.dense_2 = nn.Dense(
            features=policy_features,
            name='dense_2',
            kernel_init=initializer,
            dtype=dtype,
        )
        self.dense_3 = nn.Dense(
            features=policy_features,
            name='dense_3',
            kernel_init=initializer,
            dtype=dtype,
        )
        self.dense_4 = nn.Dense(
            features=policy_features,
            name='dense_4',
            kernel_init=initializer,
            dtype=dtype,
        )
        self.dense_5 = nn.Dense(
            features=value_features,
            name='dense_5',
            kernel_init=initializer,
            dtype=dtype,
        )
        self.dense_6 = nn.Dense(
            features=value_features,
            name='dense_6',
            kernel_init=initializer,
            dtype=dtype,
        )
        self.dense_7 = nn.Dense(
            features=value_features,
            name='dense_7',
            kernel_init=initializer,
            dtype=dtype,
        )
        self.dense_8 = nn.Dense(
            features=value_features,
            name='dense_8',
            kernel_init=initializer,
            dtype=dtype,
        )
        self.policy_layer = nn.Dense(
            features=2 * self.action_space,
            name='mean_layer',
            kernel_init=policy_layer_initializer,
            dtype=dtype,
        )
        self.value_layer = nn.Dense(
            features=1,
            name='value_layer',
            kernel_init=value_layer_initializer,
            dtype=dtype,
        )

    # Small Network:
    def model(self, x):
        # Policy Layer: Mean
        y = self.dense_1(x)
        y = nn.tanh(y)
        y = self.dense_2(y)
        y = nn.tanh(y)
        y = self.dense_3(y)
        y = nn.tanh(y)
        y = self.dense_4(y)

        # Value Layer:
        w = self.dense_5(x)
        w = nn.tanh(w)
        w = self.dense_6(w)
        w = nn.tanh(w)
        w = self.dense_7(w)
        w = nn.tanh(w)
        w = self.dense_8(w)

        # Output Layer:
        policy_output = self.policy_layer(y)
        mean, std = jnp.split(policy_output, 2, axis=-1)
        mean = nn.tanh(mean)
        std = nn.softplus(std)

        # mean = self.policy_layer(y)
        # std = 0.1 * jnp.ones_like(mean)

        values = self.value_layer(w)
        return mean, std, values

    def __call__(self, x):
        mean, std, values = self.model(x)
        return mean, std, values


class ActorCriticNetworkVmap(nn.Module):
    action_space: int

    def setup(self) -> None:
        # Shared Params:
        self.model = nn.vmap(
            ActorCriticNetwork,
            variable_axes={'params': None},
            split_rngs={'params': False},
            in_axes=0,
        )(
            action_space=self.action_space,
        )

    def __call__(self, x):
        mean, std, values = self.model(x)
        return mean, std, values
