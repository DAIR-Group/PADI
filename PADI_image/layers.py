import numpy as np
import torch
import torch.nn as nn
from ..util import USE_CUDA_FOR_PVALUE
try:
    from numba import cuda as numba_cuda
    CUDA_AVAILABLE = numba_cuda.is_available() and USE_CUDA_FOR_PVALUE
except ImportError:
    CUDA_AVAILABLE = False
    numba_cuda = None
THREADS_PER_BLOCK = 256
SI_DTYPE = np.float32

def _pair2(x):
    return (x, x) if isinstance(x, int) else tuple(x)

def _grid1d(total):
    threads = THREADS_PER_BLOCK
    blocks = (total + threads - 1) // threads
    return (blocks, threads)

def affine_eval_device(d_A, d_B, z):
    from .kernels import affine_eval_kernel
    d_X = numba_cuda.device_array(d_A.shape, dtype=SI_DTYPE)
    total = int(np.prod(d_A.shape))
    blocks, threads = _grid1d(total)
    affine_eval_kernel[blocks, threads](d_A, d_B, d_X, float(z), total)
    return d_X

def conv2d_forward_device(d_input, d_weight, d_bias, use_bias, stride=1, padding=0, dilation=1):
    from .kernels import conv2d_forward_kernel
    batch, in_c, in_h, in_w = map(int, d_input.shape)
    out_c, _, kh, kw = map(int, d_weight.shape)
    sh, sw = _pair2(stride)
    ph, pw = _pair2(padding)
    dh, dw = _pair2(dilation)
    oh = (in_h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (in_w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    d_out = numba_cuda.device_array((batch, out_c, oh, ow), dtype=SI_DTYPE)
    total = batch * out_c * oh * ow
    blocks, threads = _grid1d(total)
    conv2d_forward_kernel[blocks, threads](d_input, d_weight, d_bias, d_out, batch, in_c, in_h, in_w, out_c, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw, bool(use_bias))
    return d_out

def batchnorm2d_affine_device(d_A, d_B, d_scale, d_shift, z):
    from .kernels import batchnorm2d_affine_kernel
    shape = d_A.shape
    channels = int(shape[1])
    spatial = int(np.prod(shape[2:]))
    total = int(np.prod(shape))
    d_Ao = numba_cuda.device_array(shape, dtype=SI_DTYPE)
    d_Bo = numba_cuda.device_array(shape, dtype=SI_DTYPE)
    d_Xo = numba_cuda.device_array(shape, dtype=SI_DTYPE)
    blocks, threads = _grid1d(total)
    batchnorm2d_affine_kernel[blocks, threads](d_A, d_B, d_Ao, d_Bo, d_Xo, d_scale, d_shift, float(z), channels, spatial, total)
    return (d_Ao, d_Bo, d_Xo)

def leaky_relu_affine_device(d_X, d_A, d_B, d_itv_lo, d_itv_hi, d_valid, negative_slope=0.01):
    from .kernels import leaky_relu_affine_constraint_kernel
    total = int(np.prod(d_X.shape))
    d_Xo = numba_cuda.device_array(d_X.shape, dtype=SI_DTYPE)
    d_Ao = numba_cuda.device_array(d_A.shape, dtype=SI_DTYPE)
    d_Bo = numba_cuda.device_array(d_B.shape, dtype=SI_DTYPE)
    blocks, threads = _grid1d(total)
    leaky_relu_affine_constraint_kernel[blocks, threads](d_X, d_A, d_B, d_Xo, d_Ao, d_Bo, float(negative_slope), total, d_itv_lo, d_itv_hi, d_valid)
    return (d_Xo, d_Ao, d_Bo)

def maxpool2d_affine_device(d_X, d_A, d_B, d_itv_lo, d_itv_hi, d_valid, kernel_size=2, stride=2, padding=0):
    from .kernels import maxpool2d_affine_constraint_kernel
    batch, ch, ih, iw = map(int, d_X.shape)
    kh, kw = _pair2(kernel_size)
    sh, sw = _pair2(stride)
    ph, pw = _pair2(padding)
    oh = (ih + 2 * ph - kh) // sh + 1
    ow = (iw + 2 * pw - kw) // sw + 1
    d_Xo = numba_cuda.device_array((batch, ch, oh, ow), dtype=SI_DTYPE)
    d_Ao = numba_cuda.device_array((batch, ch, oh, ow), dtype=SI_DTYPE)
    d_Bo = numba_cuda.device_array((batch, ch, oh, ow), dtype=SI_DTYPE)
    total = batch * ch * oh * ow
    blocks, threads = _grid1d(total)
    maxpool2d_affine_constraint_kernel[blocks, threads](d_X, d_A, d_B, d_Xo, d_Ao, d_Bo, batch, ch, ih, iw, kh, kw, sh, sw, ph, pw, oh, ow, d_itv_lo, d_itv_hi, d_valid, total)
    return (d_Xo, d_Ao, d_Bo)

def linear_forward_device(d_input, d_weight, d_bias, use_bias=True, use_tiled=False):
    N, in_f = (int(d_input.shape[0]), int(d_input.shape[1]))
    out_f = int(d_weight.shape[0])
    d_out = numba_cuda.device_array((N, out_f), dtype=SI_DTYPE)
    if use_tiled and N >= 16 and (in_f >= 16) and (out_f >= 16):
        from .kernels import linear_forward_kernel_tiled
        tile = 16
        grid = ((out_f + tile - 1) // tile, (N + tile - 1) // tile)
        linear_forward_kernel_tiled[grid, (tile, tile)](d_input, d_weight, d_bias, d_out, N, in_f, out_f, bool(use_bias))
    else:
        from .kernels import linear_forward_kernel_naive
        total = N * out_f
        blocks, threads = _grid1d(total)
        linear_forward_kernel_naive[blocks, threads](d_input, d_weight, d_bias, d_out, N, in_f, out_f, bool(use_bias))
    return d_out

class NumbaSICache:

    def __init__(self, model, dtype=SI_DTYPE):
        net = model.net
        net.eval()
        self.layers = []
        for m in net.features:
            if isinstance(m, nn.Conv2d):
                w = m.weight.detach().cpu().numpy().astype(dtype)
                b = m.bias.detach().cpu().numpy().astype(dtype) if m.bias is not None else np.zeros(m.out_channels, dtype=dtype)
                self.layers.append({'type': 'conv', 'weight': numba_cuda.to_device(w), 'bias': numba_cuda.to_device(b), 'use_bias': m.bias is not None, 'stride': m.stride, 'padding': m.padding, 'dilation': m.dilation})
            elif isinstance(m, nn.BatchNorm2d):
                rm = m.running_mean.detach().cpu().numpy().astype(dtype)
                rv = m.running_var.detach().cpu().numpy().astype(dtype)
                gamma = m.weight.detach().cpu().numpy().astype(dtype) if m.affine else np.ones(m.num_features, dtype=dtype)
                beta = m.bias.detach().cpu().numpy().astype(dtype) if m.affine else np.zeros(m.num_features, dtype=dtype)
                scale = gamma / np.sqrt(rv + m.eps)
                shift = beta - rm * scale
                self.layers.append({'type': 'bn', 'scale': numba_cuda.to_device(scale.astype(dtype)), 'shift': numba_cuda.to_device(shift.astype(dtype))})
            elif isinstance(m, nn.LeakyReLU):
                self.layers.append({'type': 'lrelu', 'slope': float(m.negative_slope)})
            elif isinstance(m, nn.MaxPool2d):
                self.layers.append({'type': 'maxpool', 'kernel_size': m.kernel_size, 'stride': m.stride if m.stride is not None else m.kernel_size, 'padding': m.padding})
            else:
                raise TypeError(f'Unsupported layer: {type(m)}')
        fc = net.fc if hasattr(net, 'fc') else net.fc1
        w_fc = fc.weight.detach().cpu().numpy().astype(dtype)
        b_fc = fc.bias.detach().cpu().numpy().astype(dtype) if fc.bias is not None else np.zeros(w_fc.shape[0], dtype=dtype)
        self.fc_weight = numba_cuda.to_device(w_fc)
        self.fc_bias = numba_cuda.to_device(b_fc)
        self.fc_use_bias = fc.bias is not None
        c = getattr(model, 'c', None)
        if isinstance(c, torch.Tensor):
            self.c_np = c.detach().cpu().numpy().astype(dtype)
        else:
            self.c_np = np.asarray(c, dtype=dtype)

def get_numba_si_cache(model, force_rebuild=False):
    if force_rebuild or not hasattr(model, '_numba_si_cache_v2'):
        model._numba_si_cache_v2 = NumbaSICache(model)
    return model._numba_si_cache_v2

def get_intervals_scores_from_selection_events_cuda(X, a, b, model, R_squared, img_shape=None, alpha=0.01, zk=None, force_rebuild_cache=False):
    if zk is None:
        raise ValueError('CUDA SI v2 needs zk.')
    cache = get_numba_si_cache(model, force_rebuild=force_rebuild_cache)
    X = np.asarray(X, dtype=SI_DTYPE)
    a = np.asarray(a, dtype=SI_DTYPE).reshape(X.shape)
    b = np.asarray(b, dtype=SI_DTYPE).reshape(X.shape)
    if X.ndim == 2:
        n, d = X.shape
        if img_shape is None:
            raise ValueError('img_shape required when X is 2D.')
        C, H, W = img_shape
        A_host = a.reshape(n, C, H, W)
        B_host = b.reshape(n, C, H, W)
    elif X.ndim == 4:
        n = X.shape[0]
        A_host = a
        B_host = b
    else:
        raise ValueError(f'Unsupported X shape: {X.shape}')
    d_A = numba_cuda.to_device(np.ascontiguousarray(A_host, dtype=SI_DTYPE))
    d_B = numba_cuda.to_device(np.ascontiguousarray(B_host, dtype=SI_DTYPE))
    d_X = affine_eval_device(d_A, d_B, zk)
    d_itv_lo = numba_cuda.to_device(np.array([-np.inf], dtype=np.float64))
    d_itv_hi = numba_cuda.to_device(np.array([np.inf], dtype=np.float64))
    d_valid = numba_cuda.to_device(np.array([1], dtype=np.int32))
    for layer in cache.layers:
        t = layer['type']
        if t == 'conv':
            d_A = conv2d_forward_device(d_A, layer['weight'], layer['bias'], layer['use_bias'], stride=layer['stride'], padding=layer['padding'], dilation=layer['dilation'])
            d_B = conv2d_forward_device(d_B, layer['weight'], layer['bias'], False, stride=layer['stride'], padding=layer['padding'], dilation=layer['dilation'])
            d_X = affine_eval_device(d_A, d_B, zk)
        elif t == 'bn':
            d_A, d_B, d_X = batchnorm2d_affine_device(d_A, d_B, layer['scale'], layer['shift'], zk)
        elif t == 'lrelu':
            d_X, d_A, d_B = leaky_relu_affine_device(d_X, d_A, d_B, d_itv_lo, d_itv_hi, d_valid, negative_slope=layer['slope'])
        elif t == 'maxpool':
            d_X, d_A, d_B = maxpool2d_affine_device(d_X, d_A, d_B, d_itv_lo, d_itv_hi, d_valid, kernel_size=layer['kernel_size'], stride=layer['stride'], padding=layer['padding'])
    n = int(d_A.shape[0])
    inf = int(np.prod(d_A.shape[1:]))
    d_A_final = linear_forward_device(d_A.reshape((n, inf)), cache.fc_weight, cache.fc_bias, use_bias=cache.fc_use_bias, use_tiled=True)
    d_B_final = linear_forward_device(d_B.reshape((n, inf)), cache.fc_weight, cache.fc_bias, use_bias=False, use_tiled=True)
    A_final = d_A_final.copy_to_host()
    B_final = d_B_final.copy_to_host()
    itv_lo = float(d_itv_lo.copy_to_host()[0])
    itv_hi = float(d_itv_hi.copy_to_host()[0])
    valid = int(d_valid.copy_to_host()[0])
    if valid == 0 or itv_lo >= itv_hi:
        return (None, None, None, None, None)
    X_final = A_final + B_final * float(zk)
    c_np = cache.c_np.reshape(1, -1)
    scores = np.sum((X_final - c_np) ** 2, axis=1)
    Oz = np.flatnonzero(scores > float(R_squared)).astype(int)
    return ([itv_lo, itv_hi], A_final, B_final, scores, Oz)