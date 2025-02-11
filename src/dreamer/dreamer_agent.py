from dreamer import imag_loss, repl_loss
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import rssm

from typing import Dict, Tuple
import numpy as np
import rssm

class Agent(embodied.jax.Agent):
    """
    An agent for reinforcement learning that integrates encoding, dynamics, decoding,
    policy, value estimation, and other components.

    Parameters:
        observation_space: Space defining the structure of observations.
        action_space: Space defining the structure of actions.
        config: Configuration dictionary specifying model architecture and training parameters.
    """

    def __init__(self, observation_space: Dict[str, embodied.Space], action_space: Dict[str, embodied.Space], config: Dict):
        """
        Initialize the Agent.

        Args:
            observation_space: Dictionary mapping observation keys to their spaces.
            action_space: Dictionary mapping action keys to their spaces.
            config: Configuration dictionary for the agent.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

        # Exclude specific keys from encoder and decoder spaces
        excluded_keys = ('is_first', 'is_last', 'is_terminal', 'reward')
        encoder_space = {k: v for k, v in observation_space.items() if k not in excluded_keys}
        decoder_space = {k: v for k, v in observation_space.items() if k not in excluded_keys}

        # Initialize modules
        self.encoder = {
            'simple': rssm.Encoder,
        }[config.enc.typ](encoder_space, **config.enc[config.enc.typ], name='enc')

        self.dynamic = {
            'rssm': rssm.RSSM,
        }[config.dyn.typ](action_space, **config.dyn[config.dyn.typ], name='dyn')

        self.decoder = {
            'simple': rssm.Decoder,
        }[config.dec.typ](decoder_space, **config.dec[config.dec.typ], name='dec')

        # Helper functions
        self.feature_to_tensor = lambda x: jnp.concatenate([
            nn.cast(x['deter']),
            nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

        scalar_space = embodied.Space(np.float32, ())
        binary_space = embodied.Space(bool, (), 0, 2)

        self.reward_head = embodied.jax.MLPHead(scalar_space, **config.rewhead, name='rew')
        self.continuation_head = embodied.jax.MLPHead(binary_space, **config.conhead, name='con')

        discrete_distribution, continuous_distribution = config.policy_dist_disc, config.policy_dist_cont
        outputs = {k: discrete_distribution if v.discrete else continuous_distribution for k, v in action_space.items()}
        self.policy_head = embodied.jax.MLPHead(action_space, outputs, **config.policy, name='pol')

        self.value_head = embodied.jax.MLPHead(scalar_space, **config.value, name='val')
        self.slow_value_model = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar_space, **config.value, name='slowval'),
            source=self.value_head, **config.slowvalue)

        self.returns_normalizer = embodied.jax.Normalize(**config.retnorm, name='retnorm')
        self.value_normalizer = embodied.jax.Normalize(**config.valnorm, name='valnorm')
        self.advantage_normalizer = embodied.jax.Normalize(**config.advnorm, name='advnorm')

        self.modules = [
            self.dynamic, self.encoder, self.decoder, self.reward_head, self.continuation_head, self.policy_head, self.value_head]

        self.optimizer = embodied.jax.Optimizer(
            self.modules, self._make_optimizer(**config.opt), summary_depth=1, name='opt')

        # Loss scales
        loss_scales = self.config.loss_scales.copy()
        reconstruction_scale = loss_scales.pop('rec')
        loss_scales.update({k: reconstruction_scale for k in decoder_space})
        self.loss_scales = loss_scales

    @property
    def policy_module_names(self) -> str:
        """Regular expression matching policy-related module names."""
        return '^(enc|dyn|dec|pol)/'

    @property
    def external_space(self) -> Dict[str, embodied.Space]:
        """Define additional spaces for replay context."""
        spaces = {}
        spaces['consec'] = embodied.Space(np.int32)
        spaces['stepid'] = embodied.Space(np.uint8, 20)

        if self.config.replay_context:
            spaces.update(elements.tree.flatdict(dict(
                encoder=self.encoder.entry_space,
                dynamic=self.dynamic.entry_space,
                decoder=self.decoder.entry_space)))

        return spaces

    def initialize_policy(self, batch_size: int) -> Tuple:
        """Initialize the policy state."""
        zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
        return (
            self.encoder.initial(batch_size),
            self.dynamic.initial(batch_size),
            self.decoder.initial(batch_size),
            jax.tree.map(zeros, self.action_space))

    def initialize_training(self, batch_size: int) -> Tuple:
        """Initialize the training state."""
        return self.initialize_policy(batch_size)

    def initialize_reporting(self, batch_size: int) -> Tuple:
        """Initialize the reporting state."""
        return self.initialize_policy(batch_size)

    def policy_forward_pass(
        self,
        current_state: Tuple,
        observation: Dict[str, jnp.ndarray]
    ) -> Tuple[Tuple, Dict[str, jnp.ndarray], Dict]:
        """
        Perform a forward pass through the policy network.

        Args:
            current_state: Current state tuple (encoder, dynamic, decoder, previous_action).
            observation: Dictionary of observations.

        Returns:
            Updated state, sampled action, and output metrics.
        """
        (encoder_state, dynamic_state, decoder_state, previous_action) = current_state
        keywords = dict(training=False, single=True)
        reset_signal = observation['is_first']

        # Encoder step
        encoder_state, encoder_entry, tokens = self.encoder(
            encoder_state, observation, reset_signal, **keywords)

        # Dynamic step
        dynamic_state, dynamic_entry, features = self.dynamic.observe(
            dynamic_state, tokens, previous_action, reset_signal, **keywords)

        # Decoder step (optional)
        decoder_entry = {}
        if decoder_state:
            decoder_state, decoder_entry, reconstruction = self.decoder(
                decoder_state, features, reset_signal, **keywords)

        # Policy step
        policy_output = self.policy_head(self.feature_to_tensor(features), bdims=1)
        action = sample(policy_output)

        # Output metrics
        output = {}
        output['finite'] = elements.tree.flatdict(jax.tree.map(
            lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
            dict(obs=observation, carry=current_state, tokens=tokens, feat=features, act=action)))

        current_state = (encoder_state, dynamic_state, decoder_state, action)

        if self.config.replay_context:
            output.update(elements.tree.flatdict(dict(
                encoder=encoder_entry, dynamic=dynamic_entry, decoder=decoder_entry)))

        return current_state, action, output

    def training_step(
        self,
        current_state: Tuple,
        data: Dict[str, jnp.ndarray]
    ) -> Tuple[Tuple, Dict, Dict]:
        """
        Perform a single training step.

        Args:
            current_state: Current state tuple (encoder, dynamic, decoder, previous_action).
            data: Batch of training data.

        Returns:
            Updated state, outputs, and metrics.
        """
        current_state, observation, previous_action, step_id = self._apply_replay_context(current_state, data)

        metrics, (current_state, entries, outputs, _metrics) = self.optimizer(
            self.compute_loss, current_state, observation, previous_action, training=True, has_aux=True)

        metrics.update(_metrics)
        self.slow_value_model.update()

        outputs = {}

        if self.config.replay_context:
            updates = elements.tree.flatdict(dict(
                stepid=step_id, encoder=entries[0], dynamic=entries[1], decoder=entries[2]))
            B, T = observation['is_first'].shape
            assert all(x.shape[:2] == (B, T) for x in updates.values()), (
                (B, T), {k: v.shape for k, v in updates.items()})
            outputs['replay'] = updates

        current_state = (*current_state, {k: data[k][:, -1] for k in self.action_space})

        return current_state, outputs, metrics

    def compute_loss(
        self,
        current_state: Tuple,
        observation: Dict[str, jnp.ndarray],
        previous_action: Dict[str, jnp.ndarray],
        training: bool
    ) -> Tuple[jnp.ndarray, Tuple]:
        """
        Compute the total loss for the agent.

        Args:
            current_state: Current state tuple (encoder, dynamic, decoder).
            observation: Dictionary of observations.
            previous_action: Previous actions.
            training: Whether the model is in training mode.

        Returns:
            Total loss and auxiliary outputs.
        """
        encoder_state, dynamic_state, decoder_state = current_state
        reset_signal = observation['is_first']
        batch_size, time_steps = reset_signal.shape

        losses = {}
        metrics = {}

        # World Model
        encoder_state, encoder_entries, tokens = self.encoder(
            encoder_state, observation, reset_signal, training)

        dynamic_state, dynamic_entries, dyn_loss, representation_features, dyn_metric = self.dynamic.loss(
            dynamic_state, tokens, previous_action, reset_signal, training)

        losses.update(dyn_loss)
        metrics.update(dyn_metric)

        decoder_state, decoder_entries, reconstructions = self.decoder(
            decoder_state, representation_features, reset_signal, training)

        inputs = stop_gradient(self.feature_to_tensor(representation_features), skip=self.config.reward_grad)
        losses['reward'] = self.reward_head(inputs, 2).loss(observation['reward'])

        continuation_target = jnp.float32(~observation['is_terminal'])
        if self.config.contdisc:
            continuation_target *= 1 - 1 / self.config.horizon

        losses['continuation'] = self.continuation_head(self.feature_to_tensor(representation_features), 2).loss(continuation_target)

        for key, reconstruction in reconstructions.items():
            space, value = self.observation_space[key], observation[key]
            assert value.dtype == space.dtype, (key, space, value.dtype)
            target = jnp.float32(value) / 255 if isimage(space) else value
            losses[key] = reconstruction.loss(stop_gradient(target))

        assert all(x == (batch_size, time_steps) for x in [v.shape for v in losses.values()])

        # Imagination
        horizon_length = min(self.config.imag_last or time_steps, time_steps)
        imagination_horizon = self.config.imag_length
        starts = self.dynamic.starts(dynamic_entries, dynamic_state, horizon_length)

        policy_func = lambda feat: sample(self.policy_head(self.feature_to_tensor(feat), 1))
        _, img_features, img_previous_actions = self.dynamic.imagine(starts, policy_func, imagination_horizon, training)

        first = jax.tree.map(
            lambda x: x[:, -horizon_length:].reshape((batch_size * horizon_length, 1, *x.shape[2:])), representation_features)

        img_features = concat([stop_gradient(first, skip=self.config.ac_grads), stop_gradient(img_features)], 1)

        last_action = policy_func(jax.tree.map(lambda x: x[:, -1], img_features))
        last_action = jax.tree.map(lambda x: x[:, None], last_action)

        image_actions = concat([img_previous_actions, last_action], 1)

        assert all(x.shape[:2] == (batch_size * horizon_length, imagination_horizon + 1) for x in jax.tree.leaves(img_features))
        assert all(x.shape[:2] == (batch_size * horizon_length, imagination_horizon + 1) for x in jax.tree.leaves(image_actions))

        inputs = self.feature_to_tensor(img_features)
        loss, img_loss_out, metric = imag_loss(
            image_actions,
            self.reward_head(inputs, 2).pred(),
            self.continuation_head(inputs, 2).prob(1),
            self.policy_head(inputs, 2),
            self.value_head(inputs, 2),
            self.slow_value_model(inputs, 2),
            self.returns_normalizer, 
            self.value_normalizer, 
            self.advantage_normalizer,
            self.config.imag_loss,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon)

        losses.update({k: v.mean(1).reshape((batch_size, horizon_length)) for k, v in loss.items()})
        metrics.update(metric)

        # Replay
        if self.config.repval_loss:
            features = stop_gradient(representation_features, skip=self.config.repval_grad)
            last, terminal, reward = [observation[k] for k in ('is_last', 'is_terminal', 'reward')]

            bootstrap_values = img_loss_out['ret'][:, 0].reshape(batch_size, horizon_length)
            features, last, terminal, reward, bootstrap_values = jax.tree.map(
                lambda x: x[:, -horizon_length:], (features, last, terminal, reward, bootstrap_values))

            inputs = self.feature_to_tensor(features)
            loss, rep_loss_out, metric = repl_loss(
                last, terminal, reward, bootstrap_values,
                self.value_head(inputs, 2),
                self.slow_value_model(inputs, 2),
                self.value_normalizer,
                self.config.repl_loss,
                update=training,
                horizon=self.config.horizon)

            losses.update(loss)
            metrics.update(prefix(metric, 'reploss'))

        assert set(losses.keys()) == set(self.loss_scales.keys()), (
            sorted(losses.keys()), sorted(self.loss_scales.keys()))

        metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
        total_loss = sum([v.mean() * self.loss_scales[k] for k, v in losses.items()])

        current_state = (encoder_state, dynamic_state, decoder_state)
        entries = (encoder_entries, dynamic_entries, decoder_entries)
        outputs = {'tokens': tokens, 'representation_features': representation_features, 'losses': losses}

        return total_loss, (current_state, entries, outputs, metrics)