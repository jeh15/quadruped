import os

from absl import app, flags, logging

import jax
import jax.numpy as jnp
import numpy as np

import tf2onnx
import tensorflow as tf
from tensorflow.keras import layers

import onnxruntime as rt

from src.envs import unitree_go2_v12 as unitree_go2
from src.algorithms.ppo.load_utilities import load_policy

import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True)

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_name', None, 'Desired checkpoint folder name to load.', short_name='c',
)

def main(argv=None):
    # Load Policy:
    env = unitree_go2.UnitreeGo2Env(filename='unitree_go2/scene_mjx.xml', action_scale=0.5)
    make_policy, params, metadata = load_policy(
        checkpoint_name=FLAGS.checkpoint_name,
        environment=env,
    )
    inference_function = make_policy(params, deterministic=True)
    inference_fn = jax.jit(inference_function)

    class MLP(tf.keras.Model):
        def __init__(
            self,
            layer_sizes,
            activation=tf.nn.relu,
            kernel_init="lecun_uniform",
            activate_final=False,
            bias=True,
            layer_norm=False,
            mean_std=None,
        ):
            super().__init__()

            self.layer_sizes = layer_sizes
            self.activation = activation
            self.kernel_init = kernel_init
            self.activate_final = activate_final
            self.bias = bias
            self.layer_norm = layer_norm

            if mean_std is not None:
                self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
                self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
            else:
                self.mean = None
                self.std = None

            self.mlp_block = tf.keras.Sequential(name="MLP_0")
            for i, size in enumerate(self.layer_sizes):
                dense_layer = layers.Dense(
                    size,
                    activation=self.activation,
                    kernel_initializer=self.kernel_init,
                    name=f"dense_{i}",
                    use_bias=self.bias,
                )
                self.mlp_block.add(dense_layer)
                if self.layer_norm:
                    self.mlp_block.add(layers.LayerNormalization(name=f"layer_norm_{i}"))
            if not self.activate_final and self.mlp_block.layers:
                if hasattr(self.mlp_block.layers[-1], 'activation') and self.mlp_block.layers[-1].activation is not None:
                    self.mlp_block.layers[-1].activation = None

            self.submodules = [self.mlp_block]

        def call(self, inputs):
            if isinstance(inputs, list):
                inputs = inputs[0]
            if self.mean is not None and self.std is not None:
                print(self.mean.shape, self.std.shape)
                inputs = (inputs - self.mean) / self.std
            logits = self.mlp_block(inputs)
            loc, _ = tf.split(logits, 2, axis=-1)
            return tf.tanh(loc)

    def make_policy_network(
        param_size,
        mean_std,
        hidden_layer_sizes=[256, 256],
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        layer_norm=False,
    ):
        policy_network = MLP(
            layer_sizes=list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
            mean_std=mean_std,
        )
        return policy_network

    mean = params[0].mean["state"]
    std = params[0].std["state"]

    # Convert mean/std jax arrays to tf tensors.
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

    obs_size = env.observation_size
    act_size = env.action_size

    tf_policy_network = make_policy_network(
        param_size=act_size * 2,
        mean_std=mean_std,
        hidden_layer_sizes=metadata.network_metadata['policy_layer_size'],
        activation=tf.nn.swish,
    )

    example_input = tf.zeros((1, obs_size["state"][0]))
    example_output = tf_policy_network(example_input)
    print(example_output.shape)

    def transfer_weights(jax_params, tf_model):
        """
        Transfer weights from a JAX parameter dictionary to the TensorFlow model.

        Parameters:
        - jax_params: dict
        Nested dictionary with structure {block_name: {layer_name: {params}}}.
        For example:
        {
            'CNN_0': {
            'Conv_0': {'kernel': np.ndarray},
            'Conv_1': {'kernel': np.ndarray},
            'Conv_2': {'kernel': np.ndarray},
            },
            'MLP_0': {
            'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
            }
        }

        - tf_model: tf.keras.Model
        An instance of the adapted VisionMLP model containing named submodules and layers.
        """
        for layer_name, layer_params in jax_params.items():
            try:
                tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
            except ValueError:
                print(f"Layer {layer_name} not found in TensorFlow model.")
                continue
            if isinstance(tf_layer, tf.keras.layers.Dense):
                kernel = np.array(layer_params['kernel'])
                bias = np.array(layer_params['bias'])
                print(f"Transferring Dense layer {layer_name}, kernel shape {kernel.shape}, bias shape {bias.shape}")
                tf_layer.set_weights([kernel, bias])
            else:
                print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")

        print("Weights transferred successfully.")

    transfer_weights(params[1]['params'], tf_policy_network)

    # Example inputs for the model
    test_input = [np.ones((1, obs_size["state"][0]), dtype=np.float32)]

    # Define the TensorFlow input signature
    spec = [tf.TensorSpec(shape=(1, obs_size["state"][0]), dtype=tf.float32, name="obs")]

    tensorflow_pred = tf_policy_network(test_input)[0]
    # Build the model by calling it with example data
    print(f"Tensorflow prediction: {tensorflow_pred}")

    tf_policy_network.output_names = ['continuous_actions']

    # opset 11 matches isaac lab.
    output_path = f"go2_policy.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(tf_policy_network, input_signature=spec, opset=11, output_path=output_path)

    # Run inference with ONNX Runtime
    output_names = ['continuous_actions']
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(output_path, providers=providers)

    onnx_input = {
        'obs': np.ones((1, obs_size["state"][0]), dtype=np.float32)
    }
    # Prepare inputs for ONNX Runtime
    onnx_pred = m.run(output_names, onnx_input)[0][0]

    print("ONNX prediction:", onnx_pred)

    test_input = {
        'state': jnp.ones(obs_size["state"]),
        'privileged_state': jnp.zeros(obs_size["privileged_state"])
    }
    jax_pred, _ = inference_fn(test_input, jax.random.PRNGKey(0))
    print(jax_pred)

    print(onnx_pred.shape)
    print(tensorflow_pred.shape)
    print(jax_pred.shape)
    plt.plot(onnx_pred, label='onnx')
    plt.plot(tensorflow_pred, label='tensorflow')
    plt.plot(jax_pred, label='jax')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    app.run(main)
