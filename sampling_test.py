from absl import app

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt


def main(argv=None):
    # Example usage of the sampling function
    key = jax.random.PRNGKey(0)

    constant = 5.0
    dt = 0.02
    time_sample_fn = lambda key: jax.random.exponential(key) * constant
    sample_fn = lambda key: jnp.round(jax.random.exponential(key) * constant / dt).astype(jnp.int32)

    # Generate samples
    num_samples = 1000
    time_samples = jax.vmap(time_sample_fn)(jax.random.split(key, num_samples))
    samples = jax.vmap(sample_fn)(jax.random.split(key, num_samples))

    time_samples = np.array(time_samples)
    samples = np.array(samples)

    plt.figure(figsize=(10, 5))
    plt.hist(time_samples, bins=30, density=True, alpha=0.6, color='b')
    plt.xlabel("Time Sample Value")
    plt.ylabel("Density")
    plt.title("Histogram of Time Samples")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='g')
    plt.xlabel("Sample Value")
    plt.ylabel("Density")
    plt.title("Histogram of Samples")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    app.run(main)
