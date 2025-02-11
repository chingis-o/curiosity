import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient

from typing import Dict, List, Tuple
import jax.numpy as jnp
import nn  # Assuming nj.Module is part of a custom neural network library

class Encoder(nn.Module):
    """
    Encoder module for processing vector and image observations.

    Parameters:
        hidden_units: Number of hidden units in the MLP layers.
        normalization_type: Normalization type ('rms', 'layer', etc.).
        activation_function: Activation function ('gelu', 'relu', etc.).
        base_channels: Base number of channels for CNN layers.
        channel_multipliers: Multipliers for channel depths in CNN layers.
        mlp_layers: Number of MLP layers.
        convolution_kernel_size: Kernel size for CNN layers.
        use_symlog: Whether to apply symlog transformation to vector inputs.
        use_outer_layer: Whether to use an outermost CNN layer without pooling.
        use_strided_convolution: Whether to use strided convolutions instead of max-pooling.
    """

    hidden_units: int = 1024
    normalization_type: str = 'rms'
    activation_function: str = 'gelu'
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (2, 3, 4, 4)
    mlp_layers: int = 3
    convolution_kernel_size: int = 5
    use_symlog: bool = True
    use_outer_layer: bool = False
    use_strided_convolution: bool = False

    def __init__(self, observation_space: Dict[str, Tuple], **kwargs):
        """
        Initialize the Encoder.

        Args:
            observation_space: Dictionary mapping observation keys to their shapes.
            kwargs: Additional keyword arguments passed to submodules.
        """
        super().__init__()
        assert all(len(s.shape) <= 3 for s in observation_space.values()), "Observation space must have shapes of rank <= 3."
        self.observation_space = observation_space
        self.vector_keys = [k for k, s in observation_space.items() if len(s.shape) <= 2]
        self.image_keys = [k for k, s in observation_space.items() if len(s.shape) == 3]
        self.channel_depths = tuple(self.base_channels * mult for mult in self.channel_multipliers)
        self.kwargs = kwargs

    def process_vector_observations(self, vector_observations: Dict[str, jnp.ndarray], batch_dimensions: int) -> jnp.ndarray:
        """
        Process vector observations through a multi-layer perceptron (MLP).

        Args:
            vector_observations: Dictionary of vector observations.
            batch_dimensions: Number of batch dimensions.

        Returns:
            Encoded vector features.
        """
        squish_function = nn.symlog if self.use_symlog else lambda x: x
        concatenated_vectors = nn.DictConcat(
            {k: self.observation_space[k] for k in self.vector_keys},
            axis=1,
            squish=squish_function
        )(vector_observations)

        reshaped_vectors = concatenated_vectors.reshape((-1, *concatenated_vectors.shape[batch_dimensions:]))
        encoded_vectors = reshaped_vectors
        for i in range(self.mlp_layers):
            encoded_vectors = self.sub(f'mlp_layer_{i}', nn.Linear, self.hidden_units, **self.kwargs)(encoded_vectors)
            encoded_vectors = nn.act(self.activation_function)(
                self.sub(f'mlp_layer_{i}_normalization', nn.Norm, self.normalization_type)(encoded_vectors)
            )
        return encoded_vectors

    def process_image_observations(self, image_observations: List[jnp.ndarray], batch_dimensions: int) -> jnp.ndarray:
        """
        Process image observations through a convolutional neural network (CNN).

        Args:
            image_observations: List of image tensors.
            batch_dimensions: Number of batch dimensions.

        Returns:
            Encoded image features.
        """
        concatenated_images = jnp.concatenate(image_observations, axis=-1).astype(jnp.float32) / 255 - 0.5
        reshaped_images = concatenated_images.reshape((-1, *concatenated_images.shape[batch_dimensions:]))
        encoded_images = reshaped_images

        for i, channel_depth in enumerate(self.channel_depths):
            if self.use_outer_layer and i == 0:
                encoded_images = self.sub(f'cnn_layer_{i}', nn.Conv2D, channel_depth, self.convolution_kernel_size, **self.kwargs)(encoded_images)
            elif self.use_strided_convolution:
                encoded_images = self.sub(f'cnn_layer_{i}', nn.Conv2D, channel_depth, self.convolution_kernel_size, stride=2, **self.kwargs)(encoded_images)
            else:
                encoded_images = self.sub(f'cnn_layer_{i}', nn.Conv2D, channel_depth, self.convolution_kernel_size, **self.kwargs)(encoded_images)
                batch_size, height, width, channels = encoded_images.shape
                encoded_images = encoded_images.reshape((batch_size, height // 2, 2, width // 2, 2, channels)).max(axis=(2, 4))

            encoded_images = nn.act(self.activation_function)(
                self.sub(f'cnn_layer_{i}_normalization', nn.Norm, self.normalization_type)(encoded_images)
            )

        flattened_features = encoded_images.reshape((encoded_images.shape[0], -1))
        return flattened_features

    def forward_pass(
        self,
        carry_state: Dict,
        observations: Dict[str, jnp.ndarray],
        reset_signal: jnp.ndarray,
        is_training: bool,
        is_single_observation: bool = False
    ) -> Tuple[Dict, Dict, jnp.ndarray]:
        """
        Forward pass of the Encoder.

        Args:
            carry_state: Carry state from the previous step.
            observations: Dictionary of observations.
            reset_signal: Reset signal tensor.
            is_training: Whether the model is in training mode.
            is_single_observation: Whether the input is a single observation.

        Returns:
            Updated carry state, entries, and encoded tokens.
        """
        batch_dimensions = 1 if is_single_observation else 2
        output_features = []

        # Process vector observations
        if self.vector_keys:
            vector_observations = {k: observations[k] for k in self.vector_keys}
            vector_encoded = self.process_vector_observations(vector_observations, batch_dimensions)
            output_features.append(vector_encoded)

        # Process image observations
        if self.image_keys:
            image_observations = [observations[k] for k in sorted(self.image_keys)]
            assert all(x.dtype == jnp.uint8 for x in image_observations), "Image inputs must be uint8."
            image_encoded = self.process_image_observations(image_observations, batch_dimensions)
            output_features.append(image_encoded)

        # Combine outputs
        combined_features = jnp.concatenate(output_features, axis=-1)
        tokens = combined_features.reshape((*reset_signal.shape, *combined_features.shape[1:]))

        return carry_state, {}, tokens