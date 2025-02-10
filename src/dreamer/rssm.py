import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
stop_gradient = jax.lax.stop_gradient

class RSSM(nj.Module):

  deterministic: int = 4096
  hidden: int = 2048
  stochastic: int = 32
  classes: int = 32
  normalization: str = 'rms'
  activation_func: str = 'gelu'
  #?
  unroll: bool = False
  #?
  unimix: float = 0.01
  #?
  outscale: float = 1.0
  image_layers: int = 2
  observation_layers: int = 1
  dynamic_layers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  def __init__(self, action_space, **kw):
    assert self.deterministic % self.blocks == 0
    self.action_space = action_space
    self.kw = kw #?

  @property
  def entry_space(self):
    return dict(
        deterministic=elements.Space(np.float32, self.deterministic),
        stochastic=elements.Space(np.float32, (self.stochastic, self.classes)))

  def initial(self, bsize):
    current_state = nn.cast(dict(
        deterministic=jnp.zeros([bsize, self.deterministic], f32),
        stochastic=jnp.zeros([bsize, self.stochastic, self.classes], f32)))
    return current_state

  #?
  def truncate(self, entries, carry=None):
    assert entries['deterministic'].ndim == 3, entries['deterministic'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, current_state, tokens, action, reset, training, single=False):
    current_state, tokens, action = nn.cast((current_state, tokens, action))
    #single?
    if single:
      current_state, (entry, feature) = self._observe(
          current_state, tokens, action, reset, training)
      return current_state, entry, feature
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      current_state, (entries, feature) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          current_state, (tokens, action, reset), unroll=unroll, axis=1)
      return current_state, entries, feature

  #hidden observe?
  def _observe(self, current_state, tokens, action, reset, training):
    deterministic, stochastic, action = nn.mask(
        (current_state['deterministic'], current_state['stochastic'], action), ~reset)
    action = nn.DictConcat(self.action_space, 1)(action)
    action = nn.mask(action, ~reset)
    deterministic = self._core(deterministic, stochastic, action)
    tokens = tokens.reshape((*deterministic.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deterministic, tokens], -1)

    for i in range(self.observation_layers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.activation_func)(self.sub(f'obs{i}norm', nn.Norm, self.normalization)(x))
    
    logit = self._logit('obslogit', x)
    stochastic = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    #is it slow?
    current_state = dict(deter=deterministic, stoch=stochastic)
    feature = dict(deter=deterministic, stoch=stochastic, logit=logit)
    entry = dict(deter=deterministic, stoch=stochastic)

    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deterministic, stochastic, logit))
    return current_state, (entry, feature)

  def imagine(self, current_state, policy, length, training, single=False):
    #single?
    if single:
      action = policy(stop_gradient(current_state)) if callable(policy) else policy
      #maybe action embedding
      actemb = nn.DictConcat(self.action_space, 1)(action)
      deter = self._core(current_state['deterministic'], current_state['stochastic'], actemb)

      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      current_state = nn.cast(dict(deter=deter, stoch=stoch))
      feature = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))

      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return current_state, (feature, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        current_state, (feature, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(current_state), (), length, unroll=unroll, axis=1)
      else:
        current_state, (feature, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(current_state), nn.cast(policy), length, unroll=unroll, axis=1)
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return current_state, feature, action

  def loss(self, current_state, tokens, acts, reset, training):
    metrics = {}
    current_state, entries, feature = self.observe(current_state, tokens, acts, reset, training)
    prior = self._prior(feature['deterministic'])
    posterior = feature['logit']
    dynamic = self._dist(stop_gradient(posterior)).kl(self._dist(prior))
    representation = self._dist(posterior).kl(self._dist(stop_gradient(prior)))
    if self.free_nats:
      dynamic = jnp.maximum(dynamic, self.free_nats)
      representation = jnp.maximum(representation, self.free_nats)
    losses = {'dyn': dynamic, 'rep': representation}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(posterior).entropy().mean()
    return current_state, entries, losses, feature, metrics

  def _core(self, deter, stoch, action):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= stop_gradient(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.activation_func)(self.sub('dynin0norm', nn.Norm, self.normalization)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.activation_func)(self.sub('dynin1norm', nn.Norm, self.normalization)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.activation_func)(self.sub('dynin2norm', nn.Norm, self.normalization)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynamic_layers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deterministic, g, **self.kw)(x)
      x = nn.act(self.activation_func)(self.sub(f'dynhid{i}norm', nn.Norm, self.normalization)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deterministic, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _prior(self, feat):
    x = feat
    for i in range(self.image_layers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.activation_func)(self.sub(f'prior{i}norm', nn.Norm, self.normalization)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stochastic * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stochastic, self.classes))

  #distribution?
  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out
