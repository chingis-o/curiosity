import math

import einops
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient

from typing import Dict, List, Tuple
import math

class Decoder(nn.Module):
    """
    Decoder module for reconstructing vector and image observations.

    Parameters:
        hidden_units: Number of hidden units in the MLP layers.
        normalization_type: Normalization type ('rms', 'layer', etc.).
        activation_function: Activation function ('gelu', 'relu', etc.).
        output_scale: Scaling factor for the output layer.
        base_channels: Base number of channels for CNN layers.
        channel_multipliers: Multipliers for channel depths in CNN layers.
        mlp_layers: Number of MLP layers.
        convolution_kernel_size: Kernel size for CNN layers.
        use_symlog: Whether to apply symlog transformation to vector outputs.
        bottleneck_space: Dimensionality of the bottleneck space for spatial features.
        use_outer_layer: Whether to use an outermost CNN layer without upsampling.
        use_strided_convolution: Whether to use strided convolutions instead of upsampling.
    """

    hidden_units: int = 1024
    normalization_type: str = 'rms'
    activation_function: str = 'gelu'
    output_scale: float = 1.0
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (2, 3, 4, 4)
    mlp_layers: int = 3
    convolution_kernel_size: int = 5
    use_symlog: bool = True
    bottleneck_space: int = 8
    use_outer_layer: bool = False
    use_strided_convolution: bool = False

    def __init__(self, observation_space: Dict[str, Tuple], **kwargs):
        """
        Initialize the Decoder.

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
        self.image_depth = sum(observation_space[k].shape[-1] for k in self.image_keys)
        self.image_resolution = self.image_keys and observation_space[self.image_keys[0]].shape[:-1]
        self.kwargs = kwargs

    def forward_pass(
        self,
        carry_state: Dict,
        features: Dict[str, jnp.ndarray],
        reset_signal: jnp.ndarray,
        is_training: bool,
        is_single_observation: bool = False
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Forward pass of the Decoder.

        Args:
            carry_state: Carry state from the previous step.
            features: Dictionary of latent features.
            reset_signal: Reset signal tensor.
            is_training: Whether the model is in training mode.
            is_single_observation: Whether the input is a single observation.

        Returns:
            Updated carry state, entries, and reconstructed observations.
        """
        assert features['deter'].shape[-1] % self.bottleneck_space == 0
        kernel_size = self.convolution_kernel_size
        reconstructed_observations = {}
        batch_shape = reset_signal.shape

        # Prepare input features
        input_features = [nn.cast(features[k]) for k in ('stoch', 'deter')]
        input_features = [x.reshape((math.prod(batch_shape), -1)) for x in input_features]
        input_features = jnp.concatenate(input_features, axis=-1)

        # Reconstruct vector observations
        if self.vector_keys:
            spaces = {k: self.observation_space[k] for k in self.vector_keys}
            output_types = {
                k: 'categorical' if v.discrete else ('symlog_mse' if self.use_symlog else 'mse')
                for k, v in spaces.items()
            }
            mlp_kwargs = dict(**self.kwargs, act=self.activation_function, norm=self.normalization_type)
            x = self.sub('mlp', nn.MLP, self.mlp_layers, self.hidden_units, **mlp_kwargs)(input_features)
            x = x.reshape((*batch_shape, *x.shape[1:]))

            head_kwargs = dict(**self.kwargs, outscale=self.output_scale)
            outputs = self.sub('vector_head', embodied.jax.DictHead, spaces, output_types, **head_kwargs)(x)
            reconstructed_observations.update(outputs)

        # Reconstruct image observations
        if self.image_keys:
            upsampling_factor = 2 ** (len(self.channel_depths) - int(bool(self.use_outer_layer)))
            min_resolution = [int(x // upsampling_factor) for x in self.image_resolution]
            assert 3 <= min_resolution[0] <= 16, min_resolution
            assert 3 <= min_resolution[1] <= 16, min_resolution

            initial_shape = (*min_resolution, self.channel_depths[-1])
            if self.bottleneck_space:
                u, g = math.prod(initial_shape), self.bottleneck_space
                deterministic_features, stochastic_features = nn.cast((features['deter'], features['stoch']))

                # Process deterministic features
                deterministic_features = deterministic_features.reshape((-1, deterministic_features.shape[-1]))
                deterministic_features = self.sub('spatial_deterministic', nn.BlockLinear, u, g, **self.kwargs)(deterministic_features)
                deterministic_features = einops.rearrange(
                    deterministic_features, '... (g h w c) -> ... h w (g c)',
                    h=min_resolution[0], w=min_resolution[1], g=g)

                # Process stochastic features
                stochastic_features = stochastic_features.reshape((*stochastic_features.shape[:-2], -1))
                stochastic_features = stochastic_features.reshape((-1, stochastic_features.shape[-1]))
                stochastic_features = self.sub('spatial_stochastic_1', nn.Linear, 2 * self.hidden_units, **self.kwargs)(stochastic_features)
                stochastic_features = nn.act(self.activation_function)(
                    self.sub('spatial_stochastic_norm_1', nn.Norm, self.normalization_type)(stochastic_features)
                )
                stochastic_features = self.sub('spatial_stochastic_2', nn.Linear, initial_shape, **self.kwargs)(stochastic_features)

                # Combine deterministic and stochastic features
                combined_features = nn.act(self.activation_function)(
                    self.sub('spatial_norm', nn.Norm, self.normalization_type)(deterministic_features + stochastic_features)
                )
            else:
                combined_features = self.sub('spatial_initial', nn.Linear, initial_shape, **mlp_kwargs)(input_features)
                combined_features = nn.act(self.activation_function)(
                    self.sub('spatial_initial_norm', nn.Norm, self.normalization_type)(combined_features)
                )

            # Upsample and decode image features
            for i, channel_depth in reversed(list(enumerate(self.channel_depths[:-1]))):
                if self.use_strided_convolution:
                    conv_kwargs = dict(**self.kwargs, transp=True)
                    combined_features = self.sub(f'upsample_conv_{i}', nn.Conv2D, channel_depth, kernel_size, 2, **conv_kwargs)(combined_features)
                else:
                    combined_features = combined_features.repeat(2, -2).repeat(2, -3)
                    combined_features = self.sub(f'upsample_conv_{i}', nn.Conv2D, channel_depth, kernel_size, **self.kwargs)(combined_features)
                combined_features = nn.act(self.activation_function)(
                    self.sub(f'upsample_conv_norm_{i}', nn.Norm, self.normalization_type)(combined_features)
                )

            # Output layer for image reconstruction
            if self.use_outer_layer:
                output_kwargs = dict(**self.kwargs, outscale=self.output_scale)
                combined_features = self.sub('image_output', nn.Conv2D, self.image_depth, kernel_size, **output_kwargs)(combined_features)
            elif self.use_strided_convolution:
                output_kwargs = dict(**self.kwargs, outscale=self.output_scale, transp=True)
                combined_features = self.sub('image_output', nn.Conv2D, self.image_depth, kernel_size, 2, **output_kwargs)(combined_features)
            else:
                combined_features = combined_features.repeat(2, -2).repeat(2, -3)
                output_kwargs = dict(**self.kwargs, outscale=self.output_scale)
                combined_features = self.sub('image_output', nn.Conv2D, self.image_depth, kernel_size, **output_kwargs)(combined_features)

            combined_features = jax.nn.sigmoid(combined_features)
            combined_features = combined_features.reshape((*batch_shape, *combined_features.shape[1:]))

            # Split and process reconstructed images
            cumulative_splits = np.cumsum(
                [self.observation_space[k].shape[-1] for k in self.image_keys][:-1]
            )
            for key, output in zip(self.image_keys, jnp.split(combined_features, cumulative_splits, axis=-1)):
                output = embodied.jax.outs.MSE(output)
                output = embodied.jax.outs.Agg(output, 3, jnp.sum)
                reconstructed_observations[key] = output

        # Return results
        entries = {}
        return carry_state, entries, reconstructed_observations