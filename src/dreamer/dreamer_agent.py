import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import rssm

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3

class Agent(embodied.jax.Agent):

  def __init__(self, observation_space, action_space, config):
    self.observation_space = observation_space
    self.action_space = action_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    encoder_space = {k: v for k, v in observation_space.items() if k not in exclude}
    decoder_space = {k: v for k, v in observation_space.items() if k not in exclude}
    self.encoder = {
        'simple': rssm.Encoder,
    }[config.enc.typ](encoder_space, **config.enc[config.enc.typ], name='enc')
    self.dynamic = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](action_space, **config.dyn[config.dyn.typ], name='dyn')
    self.decoder = {
        'simple': rssm.Decoder,
    }[config.dec.typ](decoder_space, **config.dec[config.dec.typ], name='dec')

    self.feature2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.reward = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.continuation = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    descrete_distribution, continous_distribution = config.policy_dist_disc, config.policy_dist_cont
    outputs = {k: descrete_distribution if v.discrete else continous_distribution for k, v in action_space.items()}
    self.policy = embodied.jax.MLPHead(
        action_space, outputs, **config.policy, name='pol')

    self.value = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slow_value = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.value, **config.slowvalue)

    self.returns_norm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.value_norm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advantage_norm = embodied.jax.Normalize(**config.advnorm, name='advnorm')

    self.modules = [
        self.dynamic, self.encoder, self.decoder, self.reward, self.continuation, self.policy, self.value]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in decoder_space})
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.encoder.entry_space,
          dyn=self.dynamic.entry_space,
          dec=self.decoder.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.encoder.initial(batch_size),
        self.dynamic.initial(batch_size),
        self.decoder.initial(batch_size),
        jax.tree.map(zeros, self.action_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, current_state, observation, mode='train'):
    (encoder_state, dynamic_state, decoder_state, previous_action) = current_state
    kw = dict(training=False, single=True)
    reset = observation['is_first']
    encoder_state, encoder_entry, tokens = self.encoder(encoder_state, observation, reset, **kw)
    dynamic_state, dynamic_entry, features = self.dynamic.observe(
        dynamic_state, tokens, previous_action, reset, **kw)
    decoder_entry = {}
    if decoder_state:
      decoder_state, decoder_entry, reconstruction = self.decoder(decoder_state, features, reset, **kw)
    policy = self.policy(self.feature2tensor(features), bdims=1)
    action = sample(policy)
    output = {}
    output['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=observation, carry=current_state, tokens=tokens, feat=features, act=action)))
    current_state = (encoder_state, dynamic_state, decoder_state, action)
    if self.config.replay_context:
      output.update(elements.tree.flatdict(dict(
          enc=encoder_entry, dyn=dynamic_entry, dec=decoder_entry)))
    return current_state, action, output

  def train(self, current_state, data):
    current_state, observation, previous_action, stepid = self._apply_replay_context(current_state, data)
    metrics, (current_state, entries, outputs, _metrics) = self.opt(
        self.loss, current_state, observation, previous_action, training=True, has_aux=True)
    metrics.update(_metrics)
    self.slow_value.update()
    outputs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, encoder=entries[0], dynamic=entries[1], decoder=entries[2]))
      B, T = observation['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outputs['replay'] = updates
    # if self.config.replay.fracs.priority > 0:
    #   outs['replay']['priority'] = losses['model']
    current_state = (*current_state, {k: data[k][:, -1] for k in self.action_space})
    return current_state, outputs, metrics

  def loss(self, current_state, observation, previous_action, training):
    encoder_state, dynamic_state, decoder_state = current_state
    reset = observation['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    encoder_state, encoder_entries, tokens = self.encoder(
        encoder_state, observation, reset, training)
    dynamic_state, dynamic_entries, loss, representation_features, metric = self.dynamic.loss(
        dynamic_state, tokens, previous_action, reset, training)
    losses.update(loss)
    metrics.update(metric)
    decoder_state, decoder_entries, reconstructions = self.decoder(
        decoder_state, representation_features, reset, training)
    inputs = sg(self.feature2tensor(representation_features), skip=self.config.reward_grad)
    losses['rew'] = self.reward(inputs, 2).loss(observation['reward'])
    continuation = f32(~observation['is_terminal'])
    if self.config.contdisc:
      continuation *= 1 - 1 / self.config.horizon
    losses['con'] = self.continuation(self.feature2tensor(representation_features), 2).loss(continuation)
    for key, reconstruction in reconstructions.items():
      space, value = self.observation_space[key], observation[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = reconstruction.loss(sg(target))

    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dynamic.starts(dynamic_entries, dynamic_state, K)
    policy_func = lambda feat: sample(self.policy(self.feature2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dynamic.imagine(starts, policy_func, H, training)
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), representation_features)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    last_action = policy_func(jax.tree.map(lambda x: x[:, -1], imgfeat))
    last_action = jax.tree.map(lambda x: x[:, None], last_action)
    image_action = concat([imgprevact, last_action], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(image_action))
    inputs = self.feature2tensor(imgfeat)
    loss, imgloss_out, metric = imag_loss(
        image_action,
        self.reward(inputs, 2).pred(),
        self.continuation(inputs, 2).prob(1),
        self.policy(inputs, 2),
        self.value(inputs, 2),
        self.slow_value(inputs, 2),
        self.returns_norm, self.value_norm, self.advantage_norm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in loss.items()})
    metrics.update(metric)

    # Replay
    if self.config.repval_loss:
      feat = sg(representation_features, skip=self.config.repval_grad)
      last, term, rew = [observation[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inputs = self.feature2tensor(feat)
      loss, reploss_out, metric = repl_loss(
          last, term, rew, boot,
          self.value(inputs, 2),
          self.slow_value(inputs, 2),
          self.value_norm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(loss)
      metrics.update(prefix(metric, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    current_state = (encoder_state, dynamic_state, decoder_state)
    entries = (encoder_entries, dynamic_entries, decoder_entries)
    outs = {'tokens': tokens, 'repfeat': representation_features, 'losses': losses}
    return loss, (current_state, entries, outs, metrics)
