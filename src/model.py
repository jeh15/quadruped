import jax.numpy as jnp
from flax import linen as nn


class ActorCriticNetwork(nn.Module):
    action_space: int

    def setup(self):
        dtype = jnp.float64
        features = 256
        self.dense_1 = nn.Dense(
            features=features,
            name='dense_1',
            dtype=dtype,
        )
        self.dense_2 = nn.Dense(
            features=features,
            name='dense_2',
            dtype=dtype,
        )
        self.dense_3 = nn.Dense(
            features=features,
            name='dense_3',
            dtype=dtype,
        )
        self.dense_4 = nn.Dense(
            features=features,
            name='dense_4',
            dtype=dtype,
        )
        self.dense_5 = nn.Dense(
            features=features,
            name='dense_5',
            dtype=dtype,
        )
        self.dense_6 = nn.Dense(
            features=features,
            name='dense_6',
            dtype=dtype,
        )
        self.dense_7 = nn.Dense(
            features=features,
            name='dense_7',
            dtype=dtype,
        )
        self.dense_8 = nn.Dense(
            features=features,
            name='dense_8',
            dtype=dtype,
        )
        self.mean_layer = nn.Dense(
            features=self.action_space,
            name='mean_layer',
            dtype=dtype,
        )
        self.std_layer = nn.Dense(
            features=self.action_space,
            name='std_layer',
            dtype=dtype,
        )
        self.value_layer = nn.Dense(
            features=1,
            name='value_layer',
            dtype=dtype,
        )

    # Small Network:
    def model(self, x):
        # Shared Layers:
        x = self.dense_1(x)
        x = nn.elu(x)
        x = self.dense_2(x)
        x = nn.elu(x)

        # Policy Layer: Mean
        y = self.dense_3(x)
        y = nn.elu(y)
        y = self.dense_4(y)
        y = nn.elu(y)

        # Policy Layer: Standard Deviation
        z = self.dense_5(x)
        z = nn.elu(z)
        z = self.dense_6(z)
        z = nn.elu(z)

        # Value Layer:
        w = self.dense_7(x)
        w = nn.elu(w)
        w = self.dense_8(w)
        w = nn.elu(w)

        # Output Layer:
        mean = self.mean_layer(y)
        std = self.std_layer(z)
        std = nn.sigmoid(std)
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
