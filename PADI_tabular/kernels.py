import numpy as np
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
if CUDA_AVAILABLE:
    TABULAR_SI_TPB = 256
    TABULAR_SI_EPS = 1e-12
    TABULAR_SI_BIG = 1e+300

    @cuda.jit
    def tabular_linear_xab_kernel(X, A, B, W, bias, has_bias, X_out, A_out, B_out, in_features, out_features):
        j = cuda.grid(1)
        if j >= out_features:
            return
        sx, sa, sb = (0.0, 0.0, 0.0)
        for k in range(in_features):
            w = W[j, k]
            sx += X[k] * w
            sa += A[k] * w
            sb += B[k] * w
        if has_bias != 0:
            bj = bias[j]
            sx += bj
            sa += bj
        X_out[j] = sx
        A_out[j] = sa
        B_out[j] = sb

    @cuda.jit
    def tabular_batchnorm_xab_kernel(X, A, B, scale, shift, X_out, A_out, B_out, n_features):
        j = cuda.grid(1)
        if j >= n_features:
            return
        s = scale[j]
        t = shift[j]
        X_out[j] = X[j] * s + t
        A_out[j] = A[j] * s + t
        B_out[j] = B[j] * s

    @cuda.jit
    def tabular_leakyrelu_xab_interval_kernel(X, A, B, negative_slope, n_features, itv_lower, itv_upper, valid_flag):
        s_lower = cuda.shared.array(TABULAR_SI_TPB, dtype=np.float64)
        s_upper = cuda.shared.array(TABULAR_SI_TPB, dtype=np.float64)
        s_valid = cuda.shared.array(TABULAR_SI_TPB, dtype=np.int32)
        tid = cuda.threadIdx.x
        local_lower = -TABULAR_SI_BIG
        local_upper = TABULAR_SI_BIG
        local_valid = 1
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)
        while idx < n_features:
            x_val = X[idx]
            a_val = A[idx]
            b_val = B[idx]
            if x_val > 0.0:
                if abs(b_val) <= TABULAR_SI_EPS:
                    if a_val < -TABULAR_SI_EPS:
                        local_valid = 0
                else:
                    thr = -a_val / b_val
                    if b_val > 0.0:
                        if thr > local_lower:
                            local_lower = thr
                    elif thr < local_upper:
                        local_upper = thr
            else:
                if abs(b_val) <= TABULAR_SI_EPS:
                    if a_val > TABULAR_SI_EPS:
                        local_valid = 0
                else:
                    thr = -a_val / b_val
                    if b_val > 0.0:
                        if thr < local_upper:
                            local_upper = thr
                    elif thr > local_lower:
                        local_lower = thr
                X[idx] = negative_slope * x_val
                A[idx] = negative_slope * a_val
                B[idx] = negative_slope * b_val
            idx += stride
        s_lower[tid] = local_lower
        s_upper[tid] = local_upper
        s_valid[tid] = local_valid
        cuda.syncthreads()
        step = cuda.blockDim.x // 2
        while step > 0:
            if tid < step:
                if s_lower[tid + step] > s_lower[tid]:
                    s_lower[tid] = s_lower[tid + step]
                if s_upper[tid + step] < s_upper[tid]:
                    s_upper[tid] = s_upper[tid + step]
                if s_valid[tid + step] < s_valid[tid]:
                    s_valid[tid] = s_valid[tid + step]
            cuda.syncthreads()
            step //= 2
        if tid == 0:
            if s_valid[0] == 0:
                cuda.atomic.min(valid_flag, 0, 0)
            if s_lower[0] > -1e+200:
                cuda.atomic.max(itv_lower, 0, s_lower[0])
            if s_upper[0] < 1e+200:
                cuda.atomic.min(itv_upper, 0, s_upper[0])