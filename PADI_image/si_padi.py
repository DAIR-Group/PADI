import numpy as np
import torch
import mpmath as mp
from ..util import (
    TOL_LOOSE, TOL_VERY_LOOSE, INITIAL_SEARCH_SIGMA, 
    reshape_flat_to_images, pvalue_model_device, 
    compute_eta_one_vs_mean, compute_a_b_one_vs_mean,
    intersect, get_test_statistic_interval_vs_mean, 
    compute_p_value_from_intervals
)
from .conditioning import get_final_interval

def parametric_si(a, b, zk, model, R_squared, d, img_shape, alphalk=None, x_buffer=None):
    if alphalk is None:
        alphalk = 0.01
    X = a + b * zk
    X = X.reshape(1, d)
    X_images = reshape_flat_to_images(X, d, img_shape, 1)
    if x_buffer is not None:
        x_buffer.copy_(torch.from_numpy(X_images))
        X_tensor = x_buffer
    else:
        X_tensor = torch.from_numpy(X_images).float()
        if hasattr(model, 'device'):
            X_tensor = X_tensor.to(model.device)
    Oz_indices, scores, _ = model.anomaly_detection_fixed_radius(X_tensor, R_squared=R_squared)
    Oz_indices = np.asarray(Oz_indices, dtype=int)
    itv = get_final_interval(X, model, a, b, R_squared, Oz_indices, img_shape=img_shape, alphalk=alphalk)
    if not itv or len(itv) == 0:
        return (None, None)
    skz = 0
    eps_tol = 1e-06
    for interval in itv:
        lower, upper = interval
        if lower - eps_tol <= zk <= upper + eps_tol:
            skz = upper - zk
            if skz <= eps_tol:
                skz = eps_tol
            break
    if skz == 0:
        skz = eps_tol
    return (skz, Oz_indices)

def merge_close_intervals(list_zk, list_Oz, tolerance=1e-07):
    if len(list_zk) <= 1 or len(list_Oz) == 0:
        return (list_zk, list_Oz)
    merged_zk = [list_zk[0]]
    merged_Oz = []
    current_Oz = list_Oz[0]
    for i in range(1, len(list_Oz)):
        next_Oz = list_Oz[i]
        if isinstance(current_Oz, np.ndarray) and isinstance(next_Oz, np.ndarray):
            is_same = np.array_equal(current_Oz, next_Oz)
        elif current_Oz is None or next_Oz is None:
            is_same = current_Oz is None and next_Oz is None
        else:
            is_same = current_Oz == next_Oz
        if not is_same:
            merged_zk.append(list_zk[i])
            merged_Oz.append(current_Oz)
            current_Oz = next_Oz
    merged_zk.append(list_zk[-1])
    merged_Oz.append(current_Oz)
    return (merged_zk, merged_Oz)

def divide_and_conquer(a, b, threshold, model, R_squared, d, img_shape, alphalk=0.01, max_iterations=10000):
    zk = threshold[0]
    list_zk = [zk]
    list_Oz = []
    iterations = 0
    x_buffer = None
    if hasattr(model, 'device'):
        try:
            C, H, W = img_shape
            x_buffer = torch.zeros((1, C, H, W), dtype=torch.float32, device=model.device)
        except Exception:
            x_buffer = None
    step_eps = TOL_VERY_LOOSE
    SKIP_STEP = 1e-05
    consecutive_skips = 0
    MAX_SKIPS = 10000
    while zk < threshold[1] and iterations < max_iterations:
        skz, Oz = parametric_si(a, b, zk, model, R_squared, d, img_shape, alphalk, x_buffer=x_buffer)
        if skz is None or skz <= 0:
            zk = zk + SKIP_STEP
            list_zk.append(min(threshold[1], zk))
            list_Oz.append(None)
            iterations += 1
            consecutive_skips += 1
            if consecutive_skips >= MAX_SKIPS:
                break
            continue
        consecutive_skips = 0
        zk = zk + skz + step_eps
        list_zk.append(min(threshold[1], zk))
        list_Oz.append(Oz)
        iterations += 1
    list_zk, list_Oz = merge_close_intervals(list_zk, list_Oz, tolerance=TOL_LOOSE)
    if x_buffer is not None:
        del x_buffer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (list_zk, list_Oz)

def compute_p_value_one_vs_mean(X_test, X_refs, sigma, model, R_squared, img_shape, leaky_relu_slope=0.01):
    with mp.workdps(200):
        X_test_flat = np.asarray(X_test).flatten().astype(np.float64)
        X_refs_flat = np.asarray(X_refs)
        if X_refs_flat.ndim > 2:
            X_refs_flat = X_refs_flat.reshape(X_refs_flat.shape[0], -1)
        elif X_refs_flat.ndim == 1:
            X_refs_flat = X_refs_flat.reshape(1, -1)
        X_refs_flat = X_refs_flat.astype(np.float64)
        d = len(X_test_flat)
        m = len(X_refs_flat)
        eta, z_obs, sigma_z, X_ref_mean = compute_eta_one_vs_mean(X_test_flat, X_refs_flat, sigma)
        if sigma_z <= 0:
            return None
        sigma_z_sq = sigma_z ** 2
        a, b = compute_a_b_one_vs_mean(X_test_flat, X_refs_flat, sigma, eta, sigma_z_sq)
        itv = [-INITIAL_SEARCH_SIGMA * sigma_z, INITIAL_SEARCH_SIGMA * sigma_z]
        if not itv[0] <= z_obs <= itv[1]:
            itv = [-1.1 * abs(z_obs), 1.1 * abs(z_obs)]
        sign_itv = get_test_statistic_interval_vs_mean(d, m, X_test_flat, X_refs_flat, a, b, X_ref_mean)
        if sign_itv is None:
            return None
        itv = intersect(itv, sign_itv)
        if itv is None:
            return None
        a_test, b_test = (a[0:d], b[0:d])
        with pvalue_model_device(model):
            list_zk, list_Oz = divide_and_conquer(a_test.reshape(1, d), b_test.reshape(1, d), itv, model, R_squared, d=d, img_shape=img_shape, alphalk=leaky_relu_slope)
        if len(list_Oz) == 0:
            return None
        return compute_p_value_from_intervals(list_zk, list_Oz, z_obs, sigma_z, target_j=0)