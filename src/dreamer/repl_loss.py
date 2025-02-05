import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3

def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics
