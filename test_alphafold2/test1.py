import numpy as np

# import tensorflow._api.v2.compat.v1 as tf
#
# tf.compat.v1.disable_v2_behavior()
import haiku as hk
import jax

TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(.87962566103423978,
                                            dtype=np.float32)


def get_initializer_scale(initializer_name, input_shape):
    """Get Initializer for weights and scale to multiply activations by."""

    if initializer_name == 'zeros':
        w_init = hk.initializers.Constant(0.0)
    else:
        # fan-in scaling
        scale = 1.
        for channel_dim in input_shape:
            scale /= channel_dim
        if initializer_name == 'relu':
            scale *= 2

        noise_scale = scale

        stddev = np.sqrt(noise_scale)
        # Adjust stddev for truncation.
        stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
        w_init = hk.initializers.TruncatedNormal(mean=0.0, stddev=stddev)

    return w_init


w_init_ = get_initializer_scale('linear', [20])
print(type(w_init_))
print(w_init_)

import jax.numpy as jnp

# 创建一个形状为 (3, 4) 的二维数组
arr = jnp.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])
print('==')
print(arr[None])
print(arr[:, None])
print(arr[None].shape, arr[:, None].shape)
print(arr[None] + arr[:, None])
print((arr[None] + arr[:, None]).shape)
print('================================')
# 在数组的最外层添加新维度
expanded_arr = jnp.expand_dims(arr, axis=0)

print(expanded_arr.shape)  # 输出: (1, 3, 4)

# 在数组的末尾添加新维度
expanded_arr2 = jnp.expand_dims(arr, axis=-1)

print(expanded_arr2.shape)  # 输出: (3, 4, 1)

print([1] * 1 + [3])

print(';;;;;;;')
min_bin = 3.25
max_bin = 20.75
num_bins = 15

lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
print(lower_breaks)
lower_breaks = jnp.square(lower_breaks)
print(lower_breaks)

upper_breaks = jnp.concatenate([lower_breaks[1:],
                                jnp.array([1e8], dtype=jnp.float32)], axis=-1)
print(upper_breaks)


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    """Compute distogram from amino acid positions.

    Arguments:
      positions: [N_res, 3] Position coordinates.
      num_bins: The number of bins in the distogram.
      min_bin: The left edge of the first bin.
      max_bin: The left edge of the final bin. The final bin catches
          everything larger than `max_bin`.

    Returns:
      Distogram with the specified number of bins.
    """

    def squared_difference(x, y):
        return jnp.square(x - y)

    lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
    lower_breaks = jnp.square(lower_breaks)
    upper_breaks = jnp.concatenate([lower_breaks[1:],
                                    jnp.array([1e8], dtype=jnp.float32)], axis=-1)
    dist2 = jnp.sum(
        squared_difference(
            jnp.expand_dims(positions, axis=-2),
            jnp.expand_dims(positions, axis=-3)),
        axis=-1, keepdims=True)

    dgram = ((dist2 > lower_breaks).astype(jnp.float32) *
             (dist2 < upper_breaks).astype(jnp.float32))
    return dgram


dgram_from_positions(arr, num_bins, min_bin, max_bin)


class LayerNorm(hk.LayerNorm):
    """LayerNorm module.

    Equivalent to hk.LayerNorm but with different parameter shapes: they are
    always vectors rather than possibly higher-rank tensors. This makes it easier
    to change the layout whilst keep the model weight-compatible.
    """

    def __init__(self,
                 axis,
                 create_scale: bool,
                 create_offset: bool,
                 eps: float = 1e-5,
                 scale_init=None,
                 offset_init=None,
                 use_fast_variance: bool = False,
                 name=None,
                 param_axis=None):
        super().__init__(
            axis=axis,
            create_scale=False,
            create_offset=False,
            eps=eps,
            scale_init=None,
            offset_init=None,
            use_fast_variance=use_fast_variance,
            name=name,
            param_axis=param_axis)
        self._temp_create_scale = create_scale
        self._temp_create_offset = create_offset

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        is_bf16 = (x.dtype == jnp.bfloat16)
        if is_bf16:
            x = x.astype(jnp.float32)

        param_axis = self.param_axis[0] if self.param_axis else -1
        param_shape = (x.shape[param_axis],)

        param_broadcast_shape = [1] * x.ndim
        param_broadcast_shape[param_axis] = x.shape[param_axis]
        scale = None
        offset = None
        if self._temp_create_scale:
            scale = hk.get_parameter(
                'scale', param_shape, x.dtype, init=self.scale_init)
            scale = scale.reshape(param_broadcast_shape)

        if self._temp_create_offset:
            offset = hk.get_parameter(
                'offset', param_shape, x.dtype, init=self.offset_init)
            offset = offset.reshape(param_broadcast_shape)

        out = super().__call__(x, scale=scale, offset=offset)

        if is_bf16:
            out = out.astype(jnp.bfloat16)

        return out


def forward_fn(x):
    # 实例化:
    prev_msa_first_row = LayerNorm(
        axis=[-1],
        create_scale=True,
        create_offset=True,
        name='prev_msa_first_row_norm')
    return prev_msa_first_row(x)


forward = hk.transform(forward_fn)

dummy_x = jnp.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]], dtype=jnp.float32)
rng_key = jax.random.PRNGKey(42)

params = forward.init(rng=rng_key, x=dummy_x)

sample_x = jnp.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=jnp.float32)

output_1 = forward.apply(params=params, x=sample_x, rng=rng_key)
print(f'Output: {output_1}')

import jax.numpy as jnp

msa_activations = jnp.array([1, 2, 3, 4])
prev_msa_first_row = jnp.array([[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 10, 11, 12]], dtype=jnp.float32)

new_arr = prev_msa_first_row.at[0].add(msa_activations)

print(new_arr)

x = np.array([1, 2, 3, 4])
print(x[:, None] - x[None, :])
print(x[None, :] - x[:, None])

translation = np.arange(0, 15).reshape(5, 3)
print('====')
print(translation)

translation = jnp.moveaxis(translation, -1, 0)
translation = list(translation)
print('================================================================')
print(translation)

points = [jnp.expand_dims(x, axis=-2) for x in translation]
print(points)

heads = {
    'distogram': {
        'first_break': 2.3125,
        'last_break': 21.6875,
        'num_bins': 64,
        'weight': 0.3
    },
    'predicted_aligned_error': {
        # `num_bins - 1` bins uniformly space the
        # [0, max_error_bin A] range.
        # The final bin covers [max_error_bin A, +infty]
        # 31A gives bins with 0.5A width.
        'max_error_bin': 31.,
        'num_bins': 64,
        'num_channels': 128,
        'filter_by_resolution': True,
        'min_resolution': 0.1,
        'max_resolution': 3.0,
        'weight': 0.0,
    },
    'experimentally_resolved': {
        'filter_by_resolution': True,
        'max_resolution': 3.0,
        'min_resolution': 0.1,
        'weight': 0.01
    },
    'structure_module': {
        'num_layer': 8,
        'fape': {
            'clamp_distance': 10.0,
            'clamp_type': 'relu',
            'loss_unit_distance': 10.0
        },
        'angle_norm_weight': 0.01,
        'chi_weight': 0.5,
        'clash_overlap_tolerance': 1.5,
        'compute_in_graph_metrics': True,
        'dropout': 0.1,
        'num_channel': 384,
        'num_head': 12,
        'num_layer_in_transition': 3,
        'num_point_qk': 4,
        'num_point_v': 8,
        'num_scalar_qk': 16,
        'num_scalar_v': 16,
        'position_scale': 10.0,
        'sidechain': {
            'atom_clamp_distance': 10.0,
            'num_channel': 128,
            'num_residual_block': 2,
            'weight_frac': 0.5,
            'length_scale': 10.,
        },
        'structural_violation_loss_weight': 1.0,
        'violation_tolerance_factor': 12.0,
        'weight': 1.0
    },
    'predicted_lddt': {
        'filter_by_resolution': True,
        'max_resolution': 3.0,
        'min_resolution': 0.1,
        'num_bins': 50,
        'num_channels': 128,
        'weight': 0.01
    },
    'masked_msa': {
        'num_output': 23,
        'weight': 2.0
    },
}
import torch

t = torch.tensor([[1,2],[3,4]])
print(torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]])))

