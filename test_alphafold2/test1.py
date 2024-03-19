import numpy as np

# import tensorflow._api.v2.compat.v1 as tf
#
# tf.compat.v1.disable_v2_behavior()
import haiku as hk
import jax
import jax.numpy as jnp

q_weights = jnp.arange(0, 24).reshape(3, 2, 4)
q_avg = jnp.arange(0, 6).reshape(2, 3)
print(q_avg)
print(q_weights)

z = jnp.einsum('ba,ahc->bhc', q_avg, q_weights)
print(z)