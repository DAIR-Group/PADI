import numpy as np
from ..util import (
    NUMBA_AVAILABLE,
    solve_linear_inequality_geq,
    intersect,
    intersect_intervals,
    solve_quadratic_inequality_batch,
    get_test_statistic_interval_vs_mean
)
from .layers import CUDA_AVAILABLE
try:
    from numba import jit
except ImportError:

    def jit(*args, **kwargs):

        def decorator(func):
            return func
        return decorator

# ── Interval arithmetic ──────────────────────────────────────────────


# ── LeakyReLU constraint helpers ─────────────────────────────────────

@jit(nopython=True, cache=True)
def _compute_bounds_less_than_numba(a_flat, b_flat, activated_mask):
    lower_bound, upper_bound = (-np.inf, np.inf)
    for i in range(len(a_flat)):
        if not activated_mask[i]:
            u, v = (a_flat[i], b_flat[i])
            if abs(v) < 1e-16:
                if u >= 0:
                    return (np.inf, -np.inf)
            elif v > 0:
                upper_bound = min(upper_bound, -u / v)
            else:
                lower_bound = max(lower_bound, -u / v)
    return (lower_bound, upper_bound)

@jit(nopython=True, cache=True)
def _compute_bounds_geq_numba(a_flat, b_flat, activated_mask):
    lower_bound, upper_bound = (-np.inf, np.inf)
    for i in range(len(a_flat)):
        if activated_mask[i]:
            u, v = (a_flat[i], b_flat[i])
            if abs(v) < 1e-16:
                if u < 0:
                    return (np.inf, -np.inf)
            elif v > 0:
                lower_bound = max(lower_bound, -u / v)
            else:
                upper_bound = min(upper_bound, -u / v)
    return (lower_bound, upper_bound)

def _compute_bounds_geq_numpy(a_flat, b_flat, mask):
    lower_bound, upper_bound = (-np.inf, np.inf)
    a_sel, b_sel = (a_flat[mask], b_flat[mask])
    zero_v = np.abs(b_sel) < 1e-16
    if np.any(a_sel[zero_v] < 0):
        return (np.inf, -np.inf)
    pos_v = b_sel > 1e-16
    if np.any(pos_v):
        lower_bound = max(lower_bound, np.max(-a_sel[pos_v] / b_sel[pos_v]))
    neg_v = b_sel < -1e-16
    if np.any(neg_v):
        upper_bound = min(upper_bound, np.min(-a_sel[neg_v] / b_sel[neg_v]))
    return (lower_bound, upper_bound)

def _compute_bounds_less_than_numpy(a_flat, b_flat, mask):
    lower_bound, upper_bound = (-np.inf, np.inf)
    a_sel, b_sel = (a_flat[mask], b_flat[mask])
    zero_v = np.abs(b_sel) < 1e-16
    if np.any(a_sel[zero_v] >= 0):
        return (np.inf, -np.inf)
    pos_v = b_sel > 1e-16
    if np.any(pos_v):
        upper_bound = min(upper_bound, np.min(-a_sel[pos_v] / b_sel[pos_v]))
    neg_v = b_sel < -1e-16
    if np.any(neg_v):
        lower_bound = max(lower_bound, np.max(-a_sel[neg_v] / b_sel[neg_v]))
    return (lower_bound, upper_bound)

def process_leaky_relu_constraints_batch(a_flat, b_flat, X_flat, itv):
    if itv is None:
        return None
    activated = X_flat >= 0
    if NUMBA_AVAILABLE:
        l1, u1 = _compute_bounds_geq_numba(a_flat.astype(np.float64), b_flat.astype(np.float64), activated)
        l2, u2 = _compute_bounds_less_than_numba(a_flat.astype(np.float64), b_flat.astype(np.float64), activated)
    else:
        l1, u1 = _compute_bounds_geq_numpy(a_flat, b_flat, activated)
        l2, u2 = _compute_bounds_less_than_numpy(a_flat, b_flat, ~activated)
    if l1 > u1 or l2 > u2:
        return None
    new_lo = max(itv[0], l1, l2)
    new_hi = min(itv[1], u1, u2)
    return None if new_lo > new_hi else [new_lo, new_hi]

# ── MaxPool constraint helpers ───────────────────────────────────────

@jit(nopython=True, cache=True)
def _compute_maxpool_bounds_numba(a_pool, b_pool, max_indices, bs, oh, ow, ch, pool_win):
    lo, hi = (-np.inf, np.inf)
    for b in range(bs):
        for i in range(oh):
            for j in range(ow):
                for c in range(ch):
                    mi = max_indices[b, i, j, c]
                    am = a_pool[b, i, j, mi, c]
                    bm = b_pool[b, i, j, mi, c]
                    for k in range(pool_win):
                        if k != mi:
                            ad = am - a_pool[b, i, j, k, c]
                            bd = bm - b_pool[b, i, j, k, c]
                            if abs(bd) < 1e-16:
                                if ad < 0:
                                    return (np.inf, -np.inf)
                            elif bd > 0:
                                lo = max(lo, -ad / bd)
                            else:
                                hi = min(hi, -ad / bd)
    return (lo, hi)

# ── NumPy forward pass for SI (CPU fallback) ─────────────────────────

def conv2d_forward_numpy_nhwc(input_val, weight, padding=0):
    batch, h, w, in_c = input_val.shape
    out_c, _, kh, kw = weight.shape
    if padding > 0:
        input_val = np.pad(input_val, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    patches = np.lib.stride_tricks.sliding_window_view(input_val, (kh, kw), axis=(1, 2))
    b2, ho, wo, _, _, _ = patches.shape
    patches = np.moveaxis(patches, 3, -1)
    patches_flat = patches.reshape(b2, ho, wo, -1)
    wf = weight.transpose(0, 2, 3, 1).reshape(out_c, -1)
    return np.tensordot(patches_flat, wf, axes=([-1], [-1]))

def batch_norm_forward(input_val, running_mean, running_var, weight=None, bias=None, eps=0.0001):
    scale = np.sqrt(running_var + eps).reshape(1, 1, 1, -1)
    rm = running_mean.reshape(1, 1, 1, -1)
    out = (input_val - rm) / scale
    if weight is not None:
        out = out * weight.reshape(1, 1, 1, -1)
    if bias is not None:
        out = out + bias.reshape(1, 1, 1, -1)
    return out

def apply_leaky_relu_with_constraints(X, a, b, itv, alpha=0.01):
    Xf, af, bf = (X.reshape(-1), a.reshape(-1), b.reshape(-1))
    itv = process_leaky_relu_constraints_batch(af, bf, Xf, itv)
    if itv is None:
        return (None, None, None, None)
    act = Xf >= 0
    Xo = np.where(act, Xf, alpha * Xf).reshape(X.shape)
    ao = np.where(act, af, alpha * af).reshape(a.shape)
    bo = np.where(act, bf, alpha * bf).reshape(b.shape)
    return (Xo, ao, bo, itv)

def apply_maxpool_with_constraints(X, a, b, itv, pool_size=2):
    bs, hi, wi, ch = X.shape
    oH, oW = (hi // pool_size, wi // pool_size)

    def _rp(arr):
        return arr[:, :oH * pool_size, :oW * pool_size, :].reshape(bs, oH, pool_size, oW, pool_size, ch).transpose(0, 1, 3, 2, 4, 5)
    Xr, ar, br = (_rp(X), _rp(a), _rp(b))
    pw = pool_size * pool_size
    Xf = Xr.reshape(bs, oH, oW, pw, ch)
    af = ar.reshape(bs, oH, oW, pw, ch)
    bf = br.reshape(bs, oH, oW, pw, ch)
    mi = np.argmax(Xf, axis=3)
    bi_ = np.arange(bs)[:, None, None, None]
    hi_ = np.arange(oH)[None, :, None, None]
    wi_ = np.arange(oW)[None, None, :, None]
    ci_ = np.arange(ch)[None, None, None, :]
    Xp = Xf[bi_, hi_, wi_, mi, ci_]
    ap = af[bi_, hi_, wi_, mi, ci_]
    bp = bf[bi_, hi_, wi_, mi, ci_]
    if NUMBA_AVAILABLE:
        lo, hi2 = _compute_maxpool_bounds_numba(af.astype(np.float64), bf.astype(np.float64), mi, bs, oH, oW, ch, pw)
        if lo > hi2:
            return (None, None, None, None)
        new_lo, new_hi = (max(itv[0], lo), min(itv[1], hi2))
        if new_lo > new_hi:
            return (None, None, None, None)
        itv = [new_lo, new_hi]
    else:
        for b_i in range(bs):
            for oi in range(oH):
                for oj in range(oW):
                    for c in range(ch):
                        mx = mi[b_i, oi, oj, c]
                        am, bm = (af[b_i, oi, oj, mx, c], bf[b_i, oi, oj, mx, c])
                        for k in range(pw):
                            if k != mx:
                                ad = am - af[b_i, oi, oj, k, c]
                                bd = bm - bf[b_i, oi, oj, k, c]
                                itv = intersect(itv, solve_linear_inequality_geq(ad, bd))
                                if itv is None:
                                    return (None, None, None, None)
    return (Xp, ap, bp, itv)

def get_intervals_from_selection_events(X, a, b, model, img_shape=None, alpha=0.01):
    net = model.net
    net.eval()
    n, d = X.shape
    if img_shape is None:
        img_shape = (1, int(np.sqrt(d)), int(np.sqrt(d)))
    C, H, W = img_shape
    Xc = np.transpose(X.reshape(n, C, H, W), (0, 2, 3, 1))
    ac = np.transpose(a.reshape(n, C, H, W), (0, 2, 3, 1))
    bc = np.transpose(b.reshape(n, C, H, W), (0, 2, 3, 1))
    itv = [-np.inf, np.inf]
    for layer in net.features:
        lt = type(layer).__name__
        if 'Conv2d' in lt:
            w = layer.weight.detach().cpu().numpy()
            p = layer.padding[0] if hasattr(layer.padding, '__len__') else layer.padding
            Xc = conv2d_forward_numpy_nhwc(Xc, w, p)
            ac = conv2d_forward_numpy_nhwc(ac, w, p)
            bc = conv2d_forward_numpy_nhwc(bc, w, p)
        elif 'BatchNorm2d' in lt:
            rm = layer.running_mean.detach().cpu().numpy()
            rv = layer.running_var.detach().cpu().numpy()
            eps = layer.eps
            bw = layer.weight.detach().cpu().numpy() if layer.weight is not None else None
            bb = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
            Xc = batch_norm_forward(Xc, rm, rv, bw, bb, eps)
            ac = batch_norm_forward(ac, rm, rv, bw, bb, eps)
            scale = 1.0 / np.sqrt(rv + eps)
            if bw is not None:
                scale = scale * bw
            bc = bc * scale.reshape(1, 1, 1, -1)
        elif 'LeakyReLU' in lt:
            Xc, ac, bc, itv = apply_leaky_relu_with_constraints(Xc, ac, bc, itv, alpha)
            if itv is None:
                return (None, None, None)
        elif 'MaxPool2d' in lt:
            ps = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            Xc, ac, bc, itv = apply_maxpool_with_constraints(Xc, ac, bc, itv, ps)
            if itv is None:
                return (None, None, None)
    Xn = np.transpose(Xc, (0, 3, 1, 2)).reshape(n, -1)
    an = np.transpose(ac, (0, 3, 1, 2)).reshape(n, -1)
    bn = np.transpose(bc, (0, 3, 1, 2)).reshape(n, -1)
    fc = net.fc if hasattr(net, 'fc') else net.fc1
    fw = fc.weight.detach().cpu().numpy()
    return (itv, an @ fw.T, bn @ fw.T)

# ── Test statistic & final interval ──────────────────────────────────



def _decision_interval_for_score(A_j, B_j, c_np, R_squared, keep_anomalous):
    diff = A_j - c_np
    u = float(np.sum(diff ** 2) - R_squared)
    v = float(2.0 * np.sum(diff * B_j))
    w = float(np.sum(B_j ** 2))
    if keep_anomalous:
        out = solve_quadratic_inequality_batch(np.array([-u]), np.array([-v]), np.array([-w]))
    else:
        out = solve_quadratic_inequality_batch(np.array([u]), np.array([v]), np.array([w]))
    if out is None or len(out) == 0:
        return None
    return out[0]

def get_final_interval(X, model, a, b, R_squared, Oz_indices, img_shape=None, alphalk=0.01):
    n, d = X.shape
    if CUDA_AVAILABLE:
        from .layers import get_intervals_scores_from_selection_events_cuda
        zk = 0.0
        result = get_intervals_scores_from_selection_events_cuda(X, a, b, model, R_squared, img_shape=img_shape, alpha=alphalk, zk=zk, force_rebuild_cache=False)
        network_itv, A_final, B_final, scores, _ = result
        if network_itv is None:
            return None
    else:
        result = get_intervals_from_selection_events(X, a, b, model, img_shape=img_shape, alpha=alphalk)
        if result[0] is None:
            return None
        network_itv, A_final, B_final = result
    if isinstance(model.c, np.ndarray):
        c_np = model.c
    elif hasattr(model.c, 'detach'):
        c_np = model.c.detach().cpu().numpy()
    else:
        c_np = np.asarray(model.c)
    c_np = c_np.reshape(1, -1).astype(np.float64)
    all_decision_intervals = []
    Oz_set = set(Oz_indices.flatten().astype(int)) if isinstance(Oz_indices, np.ndarray) else set()
    for j in range(n):
        A_j = A_final[j:j + 1].astype(np.float64)
        B_j = B_final[j:j + 1].astype(np.float64)
        is_anomalous = j in Oz_set
        dec_itv = _decision_interval_for_score(A_j, B_j, c_np, R_squared, keep_anomalous=is_anomalous)
        if dec_itv is None:
            return None
        all_decision_intervals.append(dec_itv)
    combined = intersect_intervals(all_decision_intervals)
    if combined is None:
        return None
    final = []
    for al, ar in combined:
        itv = intersect([al, ar], network_itv)
        if itv is not None:
            final.append(tuple(itv))
    return final if final else None