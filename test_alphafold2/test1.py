import numpy as np

# import tensorflow._api.v2.compat.v1 as tf
#
# tf.compat.v1.disable_v2_behavior()
import haiku as hk
import jax
import jax.numpy as jnp

x = jnp.arange(0,24).reshape(4,6)

print(x)

y = jnp.split(x, 3, axis=-1)
for i in y:
    print(i)
p = [x,x,x]
print(sum(p))
