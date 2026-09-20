from contextlib import contextmanager
from typing import Tuple
import numpy as np
import torch
import mpmath as mp
TOL_LOOSE = 1e-06
TOL_VERY_LOOSE = 1e-06
INITIAL_SEARCH_SIGMA = 20
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):

        def decorator(func):
            return func
        return decorator
USE_CUDA_FOR_PVALUE = True

def resolve_device(device=None):
    if torch.cuda.is_available():
        return 'cuda'
    return device or 'cpu'

def _get_pvalue_torch_device():
    return 'cuda' if USE_CUDA_FOR_PVALUE and torch.cuda.is_available() else 'cpu'

@contextmanager
def pvalue_model_device(model):
    net = getattr(model, 'net', None)
    original_model_device = getattr(model, 'device', None)
    original_net_device = None
    if net is not None:
        try:
            original_net_device = next(net.parameters()).device
        except StopIteration:
            original_net_device = None
    target_device = _get_pvalue_torch_device()
    if net is not None:
        model.net = model.net.to(target_device).float()
    if hasattr(model, 'device'):
        model.device = target_device
    try:
        yield target_device
    finally:
        if net is not None and original_net_device is not None:
            model.net = model.net.to(original_net_device).float()
        if original_model_device is not None:
            model.device = original_model_device

def reshape_flat_to_images(X_flat: np.ndarray, d: int, img_shape: Tuple[int, int, int], n: int=None) -> np.ndarray:
    if n is None:
        n = X_flat.shape[0]
    c, h, w = img_shape
    return X_flat.reshape(n, c, h, w)

def extract_patches_tensor(img_tensor, patch_size=30, stride=30):
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    B, C, H, W = img_tensor.shape
    patches = img_tensor.unfold(2, patch_size, stride)
    patches = patches.unfold(3, patch_size, stride)
    n_h = patches.shape[2]
    n_w = patches.shape[3]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(-1, C, patch_size, patch_size)
    return patches

def compute_eta_one_vs_mean(X_test, X_refs, sigma):
    X_test = np.asarray(X_test).flatten().astype(np.float64)
    X_refs = np.asarray(X_refs).astype(np.float64)
    if X_refs.ndim == 1:
        X_refs = X_refs.reshape(1, -1)
    d = len(X_test)
    m = len(X_refs)
    n = m + 1
    X_ref_mean = np.mean(X_refs, axis=0)
    diff = X_test - X_ref_mean
    signs = np.ones(d, dtype=np.float64)
    signs[diff < 0] = -1.0
    eta = np.zeros(n * d, dtype=np.float64)
    eta[0:d] = signs
    eta[d:] = np.tile(-signs / m, m)
    X_flat = np.concatenate([X_test, X_refs.flatten()])
    z_obs = eta.dot(X_flat)
    if np.isscalar(sigma):
        etaTsigmaeta = (1 + 1 / m) * d * sigma
    elif isinstance(sigma, np.ndarray) and sigma.ndim == 1:
        etaTsigmaeta = (1 + 1 / m) * np.sum(sigma)
    elif isinstance(sigma, np.ndarray) and sigma.ndim == 2:
        eta_base = signs.reshape(-1, 1)
        etaTsigmaeta = (1 + 1 / m) * float((eta_base.T @ sigma @ eta_base).item())
    else:
        raise ValueError(f'Unsupported sigma type: {type(sigma)}')
    sigma_z = np.sqrt(etaTsigmaeta)
    return (eta, z_obs, sigma_z, X_ref_mean)

def compute_a_b_one_vs_mean(X_test, X_refs, sigma, eta, sigma_z_squared):
    X_test = np.asarray(X_test).flatten().astype(np.float64)
    X_refs = np.asarray(X_refs).astype(np.float64)
    if X_refs.ndim == 1:
        X_refs = X_refs.reshape(1, -1)
    d = len(X_test)
    m = len(X_refs)
    n = m + 1
    signs = eta[0:d].copy()
    if np.isscalar(sigma):
        Sigma_signs_test = sigma * signs
    elif isinstance(sigma, np.ndarray) and sigma.ndim == 1:
        Sigma_signs_test = sigma * signs
    elif isinstance(sigma, np.ndarray) and sigma.ndim == 2:
        Sigma_signs_test = sigma @ signs
    else:
        raise ValueError(f'Unsupported sigma type: {type(sigma)}')
    b = np.zeros(n * d, dtype=np.float64)
    b[0:d] = Sigma_signs_test / sigma_z_squared
    for k in range(m):
        b[(k + 1) * d:(k + 2) * d] = -Sigma_signs_test / (m * sigma_z_squared)
    X_flat = np.concatenate([X_test, X_refs.flatten()])
    z_obs = eta.dot(X_flat)
    a = X_flat - z_obs * b
    return (a, b)

# ── Shared Interval & Math Utilities ──────────────────────────────────

def solve_linear_inequality_geq(a, b):
    a, b = float(a), float(b)
    if abs(b) < 1e-16:
        return [-np.inf, np.inf] if a >= 0 else None
    if b > 0:
        return [-a / b, np.inf]
    return [-np.inf, -a / b]

def solve_linear_inequality(u, v):
    u, v = float(u), float(v)
    if abs(v) < 1e-16:
        return [-np.inf, np.inf] if u <= 0 else None
    if v > 0:
        return [-np.inf, -u / v]
    return [-u / v, np.inf]

def intersect(itv1, itv2):
    if itv1 is None or itv2 is None:
        return None
    itv = [max(itv1[0], itv2[0]), min(itv1[1], itv2[1])]
    return None if itv[0] > itv[1] else itv

def intersect_intervals(intervals_list):
    if any((x is None for x in intervals_list)):
        return None
    if len(intervals_list) == 0:
        return [(-np.inf, np.inf)]
    
    result = []
    for itv in intervals_list[0]:
        if isinstance(itv, (list, tuple)) and len(itv) == 2:
            result.append((itv[0], itv[1]))
        else:
            result = list(intervals_list[0])
            break
            
    for intervals in intervals_list[1:]:
        new_result = []
        for a1, b1 in result:
            for item in intervals:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    a2, b2 = item[0], item[1]
                else:
                    continue
                lower = max(a1, a2)
                upper = min(b1, b2)
                if lower < upper:
                    new_result.append((lower, upper))
        result = new_result
        if not result:
            return None
    return result

def solve_quadratic_inequality_batch(u, v, w):
    n = len(u)
    intervals = []
    for i in range(n):
        ui, vi, wi = u[i], v[i], w[i]
        if abs(wi) < 1e-16:
            if abs(vi) < 1e-16:
                if ui < 0:
                    intervals.append([(-np.inf, np.inf)])
                else:
                    return None
            elif vi > 0:
                intervals.append([(-np.inf, -ui / vi)])
            else:
                intervals.append([(-ui / vi, np.inf)])
            continue
        delta = vi ** 2 - 4 * wi * ui
        if delta < 0:
            if wi < 0:
                intervals.append([(-np.inf, np.inf)])
            else:
                return None
        elif abs(delta) < 1e-16:
            if abs(wi) < 1e-14:
                if abs(vi) < 1e-16:
                    if ui < 0:
                        intervals.append([(-np.inf, np.inf)])
                    else:
                        return None
                elif vi > 0:
                    intervals.append([(-np.inf, -ui / vi)])
                else:
                    intervals.append([(-ui / vi, np.inf)])
                continue
            z0 = -vi / (2 * wi)
            if wi < 0:
                intervals.append([(-np.inf, z0), (z0, np.inf)])
            else:
                return None
        else:
            sqrt_delta = np.sqrt(delta)
            z1 = (-vi - sqrt_delta) / (2 * wi)
            z2 = (-vi + sqrt_delta) / (2 * wi)
            if z1 > z2:
                z1, z2 = z2, z1
            if wi < 0:
                intervals.append([(-np.inf, z1), (z2, np.inf)])
            else:
                intervals.append([(z1, z2)])
    return intervals

def get_test_statistic_interval_vs_mean(d, m, X_test, X_refs, a, b, X_ref_mean):
    X_test = np.asarray(X_test).flatten()
    X_refs = np.asarray(X_refs)
    if X_refs.ndim == 1:
        X_refs = X_refs.reshape(1, -1)
    X_ref_mean = np.asarray(X_ref_mean).flatten()
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    
    a_test = a[0:d]
    b_test = b[0:d]
    a_refs_mean = np.zeros(d)
    b_refs_mean = np.zeros(d)
    for k in range(m):
        a_refs_mean += a[(k + 1) * d:(k + 2) * d]
        b_refs_mean += b[(k + 1) * d:(k + 2) * d]
    a_refs_mean /= m
    b_refs_mean /= m
    
    itv = [-float('inf'), float('inf')]
    for i in range(d):
        diff = X_test[i] - X_ref_mean[i]
        a_diff = a_test[i] - a_refs_mean[i]
        b_diff = b_test[i] - b_refs_mean[i]
        if diff >= 0:
            constraint = solve_linear_inequality_geq(a_diff, b_diff)
        else:
            constraint = solve_linear_inequality(a_diff, b_diff)
        itv = intersect(itv, constraint)
        if itv is None:
            return None
    return itv

def compute_p_value_from_intervals(list_zk, list_Oz, z_obs, sigma_z, target_j=0):
    mp.dps = 200
    if len(list_zk) == 0:
        return None
    try:
        sigma_mp = mp.mpf(sigma_z)
        z_obs_mp = mp.mpf(z_obs)
        numer = mp.mpf(0)
        denom = mp.mpf(0)
    except Exception:
        return None
        
    for i in range(len(list_zk) - 1):
        Oz = list_Oz[i]
        if Oz is None:
            continue
            
        is_target = False
        if hasattr(Oz, '__iter__'):
            oz_set = set(Oz.flatten().astype(int)) if isinstance(Oz, np.ndarray) else set(Oz)
            is_target = target_j in oz_set
        else:
            is_target = target_j == Oz
            
        if not is_target:
            continue
            
        al = mp.mpf(list_zk[i])
        ar = mp.mpf(list_zk[i + 1])
        prob = mp.ncdf(ar / sigma_mp) - mp.ncdf(al / sigma_mp)
        if prob < 0:
            prob = mp.mpf(0)
        denom += prob
        
        if z_obs_mp >= ar:
            numer += prob
        elif z_obs_mp >= al:
            term = mp.ncdf(z_obs_mp / sigma_mp) - mp.ncdf(al / sigma_mp)
            if term < 0:
                term = mp.mpf(0)
            numer += term
            
    if denom == 0:
        normalized_z = abs(z_obs / sigma_z)
        if normalized_z > 15:
            return 0.0
        return None
        
    p = mp.mpf(1) - numer / denom
    return float(max(mp.mpf(0), min(mp.mpf(1), p)))