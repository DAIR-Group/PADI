import numpy as np
import torch
import torch.nn as nn
import mpmath as mp
from ..util import (
    compute_eta_one_vs_mean, 
    compute_a_b_one_vs_mean,
    solve_linear_inequality_geq, 
    intersect, 
    solve_linear_inequality, 
    get_test_statistic_interval_vs_mean,
    compute_p_value_from_intervals
)
from .conditioning import get_final_interval
from .kernels import CUDA_AVAILABLE
if CUDA_AVAILABLE:
    from numba import cuda
    from .kernels import tabular_linear_xab_kernel, tabular_batchnorm_xab_kernel, tabular_leakyrelu_xab_interval_kernel
TABULAR_SI_TPB = 256

def _extract_net(model_or_net):
    if hasattr(model_or_net, 'net') and model_or_net.net is not None:
        return model_or_net.net
    return model_or_net

def _to_np64(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)

class _SICache:

    def __init__(self, model_or_net):
        self.net = _extract_net(model_or_net)
        self.layers = []
        self._build()

    def _build(self):
        net = self.net
        for layer in net.features:
            if isinstance(layer, nn.Linear):
                self._add_linear(layer)
            elif isinstance(layer, nn.BatchNorm1d):
                self._add_bn(layer)
            elif isinstance(layer, nn.LeakyReLU):
                self.layers.append({'type': 'leakyrelu', 'slope': float(layer.negative_slope)})
            else:
                raise TypeError(f'Unsupported layer: {type(layer)}')
        self._add_linear(net.fc)

    def _add_linear(self, layer):
        W = _to_np64(layer.weight)
        bias = _to_np64(layer.bias).reshape(-1) if layer.bias is not None else np.zeros(W.shape[0], dtype=np.float64)
        item = {'type': 'linear', 'W_h': W, 'bias_h': bias, 'has_bias': int(layer.bias is not None), 'in_f': W.shape[1], 'out_f': W.shape[0]}
        if CUDA_AVAILABLE:
            item['W_d'] = cuda.to_device(W)
            item['bias_d'] = cuda.to_device(bias)
        self.layers.append(item)

    def _add_bn(self, layer):
        rm = _to_np64(layer.running_mean)
        rv = _to_np64(layer.running_var)
        w = _to_np64(layer.weight) if layer.affine else np.ones_like(rm)
        b = _to_np64(layer.bias) if layer.affine else np.zeros_like(rm)
        scale = (w / np.sqrt(rv + layer.eps)).astype(np.float64)
        shift = (b - rm * scale).astype(np.float64)
        item = {'type': 'batchnorm', 'scale_h': scale, 'shift_h': shift, 'n_f': scale.size}
        if CUDA_AVAILABLE:
            item['scale_d'] = cuda.to_device(scale)
            item['shift_d'] = cuda.to_device(shift)
        self.layers.append(item)
_CACHE = {}

def invalidate_tabular_si_cache(model_or_net=None):
    global _CACHE
    if model_or_net is None:
        _CACHE = {}
    else:
        _CACHE.pop(id(_extract_net(model_or_net)), None)

def _get_cache(model, force=False):
    net = _extract_net(model)
    key = id(net)
    if force or key not in _CACHE:
        _CACHE[key] = _SICache(net)
    return _CACHE[key]

def _propagate_cpu(X, A, B, cache):
    itv = [-float('inf'), float('inf')]
    X_c, A_c, B_c = (X.copy(), A.copy(), B.copy())
    for L in cache.layers:
        t = L['type']
        if t == 'linear':
            W, bias = (L['W_h'], L['bias_h'])
            X_c, A_c, B_c = (W @ X_c, W @ A_c, W @ B_c)
            if L['has_bias']:
                X_c += bias
                A_c += bias
        elif t == 'batchnorm':
            s, sh = (L['scale_h'], L['shift_h'])
            X_c, A_c, B_c = (X_c * s + sh, A_c * s + sh, B_c * s)
        elif t == 'leakyrelu':
            sl = L['slope']
            for i in range(X_c.size):
                c = solve_linear_inequality_geq(A_c[i], B_c[i]) if X_c[i] > 0 else solve_linear_inequality(A_c[i], B_c[i])
                itv = intersect(itv, c)
                if itv is None:
                    return (None, A_c, B_c, X_c)
            mask = X_c > 0
            X_c = np.where(mask, X_c, sl * X_c)
            A_c = np.where(mask, A_c, sl * A_c)
            B_c = np.where(mask, B_c, sl * B_c)
    return (itv, A_c, B_c, X_c)

def _launch(n):
    tpb = TABULAR_SI_TPB
    return ((int((n + tpb - 1) // tpb),), (int(tpb),))

def propagate_xab(X, a, b, model, zk=None, force_rebuild_cache=False):
    cache = _get_cache(model, force=force_rebuild_cache)
    A0 = np.asarray(a, dtype=np.float64).reshape(-1)
    B0 = np.asarray(b, dtype=np.float64).reshape(-1)
    X0 = A0 + B0 * float(zk) if zk is not None else np.asarray(X, dtype=np.float64).reshape(-1)
    if not CUDA_AVAILABLE:
        return _propagate_cpu(X0, A0, B0, cache)
    X_d = cuda.to_device(X0)
    A_d = cuda.to_device(A0)
    B_d = cuda.to_device(B0)
    lo_d = cuda.to_device(np.array([-np.inf], dtype=np.float64))
    hi_d = cuda.to_device(np.array([np.inf], dtype=np.float64))
    vl_d = cuda.to_device(np.array([1], dtype=np.int32))
    for L in cache.layers:
        t = L['type']
        if t == 'linear':
            of = L['out_f']
            Xo = cuda.device_array(of, dtype=np.float64)
            Ao = cuda.device_array(of, dtype=np.float64)
            Bo = cuda.device_array(of, dtype=np.float64)
            bk, th = _launch(of)
            tabular_linear_xab_kernel[bk, th](X_d, A_d, B_d, L['W_d'], L['bias_d'], np.int32(L['has_bias']), Xo, Ao, Bo, np.int32(L['in_f']), np.int32(of))
            X_d, A_d, B_d = (Xo, Ao, Bo)
        elif t == 'batchnorm':
            nf = L['n_f']
            Xo = cuda.device_array(nf, dtype=np.float64)
            Ao = cuda.device_array(nf, dtype=np.float64)
            Bo = cuda.device_array(nf, dtype=np.float64)
            bk, th = _launch(nf)
            tabular_batchnorm_xab_kernel[bk, th](X_d, A_d, B_d, L['scale_d'], L['shift_d'], Xo, Ao, Bo, np.int32(nf))
            X_d, A_d, B_d = (Xo, Ao, Bo)
        elif t == 'leakyrelu':
            nf = X_d.shape[0]
            bk, th = _launch(nf)
            tabular_leakyrelu_xab_interval_kernel[bk, th](X_d, A_d, B_d, float(L['slope']), np.int32(nf), lo_d, hi_d, vl_d)
            if int(vl_d.copy_to_host()[0]) == 0:
                return (None, None, None, None)
            lo = float(lo_d.copy_to_host()[0])
            hi = float(hi_d.copy_to_host()[0])
            if lo >= hi:
                return (None, None, None, None)
    lo = float(lo_d.copy_to_host()[0])
    hi = float(hi_d.copy_to_host()[0])
    if lo >= hi:
        return (None, None, None, None)
    return ([lo, hi], A_d.copy_to_host(), B_d.copy_to_host(), X_d.copy_to_host())



def parametric_si(a, b, zk, model, R_squared, center_c, d, alpha=0.01, force_rebuild_cache=False):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    final_intervals, Oz = get_final_interval(None, model, a.reshape(1, -1), b.reshape(1, -1), R_squared, center_c, alpha=alpha, zk=zk, force_rebuild_cache=force_rebuild_cache)
    if final_intervals is None or len(final_intervals) == 0:
        return (None, Oz, None)
    for lower, upper in final_intervals:
        if lower <= zk <= upper:
            step = upper - zk
            return (max(step, 1e-06), Oz, final_intervals)
    return (None, Oz, final_intervals)

def divide_and_conquer(a, b, threshold, model, R_squared, center_c, d, alpha=0.01, max_iterations=10000, force_rebuild_cache=False):
    zk = float(threshold[0])
    z_right = float(threshold[1])
    list_zk = [zk]
    list_Oz = []
    it = 0
    SKIP_STEP = 1e-05
    consecutive_skips = 0
    MAX_SKIPS = 10000
    while zk < z_right and it < max_iterations:
        it += 1
        skz, Oz, _ = parametric_si(a, b, zk, model, R_squared, center_c, d, alpha=alpha, force_rebuild_cache=force_rebuild_cache)
        if skz is None or skz <= 0:
            zk += SKIP_STEP
            list_zk.append(min(z_right, zk))
            list_Oz.append(None)
            consecutive_skips += 1
            if consecutive_skips >= MAX_SKIPS:
                break
            continue
        consecutive_skips = 0
        zk = min(z_right, zk + skz + 1e-06) if not np.isinf(skz) else z_right
        list_zk.append(zk)
        list_Oz.append(Oz)
    return (list_zk, list_Oz)

def compute_p_value_one_vs_mean(X_test, X_refs, sigma, model, R_squared, center_c, leaky_relu_slope=0.01, return_failure_reason=False, force_rebuild_cache=False):

    def _ret(p, r):
        return (p, r) if return_failure_reason else p
    with mp.workdps(200):
        X_test = np.asarray(X_test).flatten().astype(np.float64)
        X_refs = np.asarray(X_refs).astype(np.float64)
        if X_refs.ndim == 1:
            X_refs = X_refs.reshape(1, -1)
        d, m = (len(X_test), len(X_refs))
        eta, z_obs, sigma_z, X_ref_mean = compute_eta_one_vs_mean(X_test, X_refs, sigma)
        if sigma_z <= 0:
            return _ret(None, 'sigma_z <= 0')
        a, b = compute_a_b_one_vs_mean(X_test, X_refs, sigma, eta, sigma_z ** 2)
        itv = [-20 * sigma_z, 20 * sigma_z]
        if not itv[0] <= z_obs <= itv[1]:
            itv = [-1.25 * abs(z_obs), 1.25 * abs(z_obs)]
        sign_itv = get_test_statistic_interval_vs_mean(d, m, X_test, X_refs, a, b, X_ref_mean)
        if sign_itv is None:
            return _ret(None, 'sign_interval_empty')
        itv = intersect(itv, sign_itv)
        if itv is None:
            return _ret(None, 'interval_intersection_empty')
        a_test, b_test = (a[0:d], b[0:d])
        list_zk, list_Oz = divide_and_conquer(a_test, b_test, itv, model, R_squared, center_c, d=d, alpha=leaky_relu_slope, force_rebuild_cache=force_rebuild_cache)
        if len(list_zk) < 2:
            return _ret(None, 'divide_conquer_empty')
        p = compute_p_value_from_intervals(list_zk, list_Oz, z_obs, sigma_z, target_j=0)
        if p is None:
            return _ret(None, 'p_value_computation_failed')
        return _ret(p, None)

def evaluate_si(model, X_test, X_refs, sigma, return_failure_reason=False):
    X_test = np.asarray(X_test).astype(np.float64)
    X_refs = np.asarray(X_refs).astype(np.float64)
    if X_test.ndim > 1:
        X_test = X_test.flatten()
    if X_refs.ndim == 1:
        X_refs = X_refs.reshape(1, -1)
    net = _extract_net(model)
    net.eval()
    if hasattr(model, 'c') and model.c is not None:
        center_c = _to_np64(model.c).reshape(-1)
        R_squared = model.R_squared if model.R_squared is not None else 1.0
    else:
        center_c = np.zeros(net.fc.out_features, dtype=np.float64)
        R_squared = 1.0
    return compute_p_value_one_vs_mean(X_test, X_refs, sigma, net, R_squared, center_c, return_failure_reason=return_failure_reason)