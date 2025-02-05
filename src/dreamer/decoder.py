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

class Decoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training, single=False):
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=minres[0], w=minres[1], g=g)
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = embodied.jax.outs.MSE(out)
        out = embodied.jax.outs.Agg(out, 3, jnp.sum)
        recons[k] = out

    entries = {}
    return carry, entries, recons