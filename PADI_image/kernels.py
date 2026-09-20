import numpy as np
from numba import cuda, float32
THREADS_PER_BLOCK = 256
SI_DTYPE = np.float32
SI_EPS = 1e-12
SI_INF = 1e+300

@cuda.jit
def affine_eval_kernel(A, B, X, z, total):
    idx = cuda.grid(1)
    while idx < total:
        X.flat[idx] = A.flat[idx] + B.flat[idx] * z
        idx += cuda.gridsize(1)

@cuda.jit
def conv2d_forward_kernel(input_data, weight, bias, output, batch, in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, use_bias):
    idx = cuda.grid(1)
    total_elements = batch * out_channels * out_height * out_width
    while idx < total_elements:
        ow = idx % out_width
        temp = idx // out_width
        oh = temp % out_height
        temp = temp // out_height
        oc = temp % out_channels
        b = temp // out_channels
        sum_val = 0.0
        for ic in range(in_channels):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    ih = oh * stride_h - pad_h + kh * dilation_h
                    iw = ow * stride_w - pad_w + kw * dilation_w
                    if 0 <= ih < in_height and 0 <= iw < in_width:
                        input_idx = b * (in_channels * in_height * in_width) + ic * (in_height * in_width) + ih * in_width + iw
                        weight_idx = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w) + kh * kernel_w + kw
                        sum_val += input_data.flat[input_idx] * weight.flat[weight_idx]
        if use_bias:
            sum_val += bias[oc]
        output.flat[idx] = sum_val
        idx += cuda.gridsize(1)

@cuda.jit
def batchnorm2d_affine_kernel(A_in, B_in, A_out, B_out, X_out, scale, shift, z, channels, spatial_size, total):
    idx = cuda.grid(1)
    while idx < total:
        c = idx // spatial_size % channels
        a = A_in.flat[idx] * scale[c] + shift[c]
        b_val = B_in.flat[idx] * scale[c]
        A_out.flat[idx] = a
        B_out.flat[idx] = b_val
        X_out.flat[idx] = a + b_val * z
        idx += cuda.gridsize(1)

@cuda.jit
def leaky_relu_affine_constraint_kernel(X_in, A_in, B_in, X_out, A_out, B_out, negative_slope, total, itv_lower, itv_upper, valid_flag):
    s_lower = cuda.shared.array(THREADS_PER_BLOCK, dtype=np.float64)
    s_upper = cuda.shared.array(THREADS_PER_BLOCK, dtype=np.float64)
    tid = cuda.threadIdx.x
    local_lower = -SI_INF
    local_upper = SI_INF
    local_valid = 1
    idx = cuda.grid(1)
    while idx < total:
        x_val = X_in.flat[idx]
        a_val = A_in.flat[idx]
        b_val = B_in.flat[idx]
        if x_val >= 0.0:
            A_out.flat[idx] = a_val
            B_out.flat[idx] = b_val
            X_out.flat[idx] = x_val
            if abs(b_val) <= SI_EPS:
                if a_val < -SI_EPS:
                    local_valid = 0
            else:
                threshold = -a_val / b_val
                if b_val > 0.0:
                    if threshold > local_lower:
                        local_lower = threshold
                elif threshold < local_upper:
                    local_upper = threshold
        else:
            A_out.flat[idx] = negative_slope * a_val
            B_out.flat[idx] = negative_slope * b_val
            X_out.flat[idx] = negative_slope * x_val
            if abs(b_val) <= SI_EPS:
                if a_val > SI_EPS:
                    local_valid = 0
            else:
                threshold = -a_val / b_val
                if b_val > 0.0:
                    if threshold < local_upper:
                        local_upper = threshold
                elif threshold > local_lower:
                    local_lower = threshold
        idx += cuda.gridsize(1)
    s_lower[tid] = local_lower
    s_upper[tid] = local_upper
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            if s_lower[tid + stride] > s_lower[tid]:
                s_lower[tid] = s_lower[tid + stride]
            if s_upper[tid + stride] < s_upper[tid]:
                s_upper[tid] = s_upper[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        if local_valid == 0:
            cuda.atomic.min(valid_flag, 0, 0)
        if s_lower[0] > -1e+200:
            cuda.atomic.max(itv_lower, 0, s_lower[0])
        if s_upper[0] < 1e+200:
            cuda.atomic.min(itv_upper, 0, s_upper[0])

@cuda.jit
def maxpool2d_affine_constraint_kernel(X_in, A_in, B_in, X_out, A_out, B_out, batch, channels, in_height, in_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, out_height, out_width, itv_lower, itv_upper, valid_flag, total_out):
    s_lower = cuda.shared.array(THREADS_PER_BLOCK, dtype=np.float64)
    s_upper = cuda.shared.array(THREADS_PER_BLOCK, dtype=np.float64)
    tid = cuda.threadIdx.x
    local_lower = -SI_INF
    local_upper = SI_INF
    local_valid = 1
    idx = cuda.grid(1)
    while idx < total_out:
        ow = idx % out_width
        tmp = idx // out_width
        oh = tmp % out_height
        tmp = tmp // out_height
        c = tmp % channels
        n = tmp // channels
        h_start = oh * stride_h - pad_h
        w_start = ow * stride_w - pad_w
        h_end = h_start + kernel_h
        w_end = w_start + kernel_w
        if h_start < 0:
            h_start = 0
        if w_start < 0:
            w_start = 0
        if h_end > in_height:
            h_end = in_height
        if w_end > in_width:
            w_end = in_width
        max_val = -SI_INF
        max_idx = -1
        for h in range(h_start, h_end):
            for w_pos in range(w_start, w_end):
                curr_idx = n * (channels * in_height * in_width) + c * (in_height * in_width) + h * in_width + w_pos
                val = X_in.flat[curr_idx]
                if val > max_val:
                    max_val = val
                    max_idx = curr_idx
        if max_idx == -1:
            local_valid = 0
        else:
            a_max = A_in.flat[max_idx]
            b_max = B_in.flat[max_idx]
            A_out.flat[idx] = a_max
            B_out.flat[idx] = b_max
            X_out.flat[idx] = max_val
            for h in range(h_start, h_end):
                for w_pos in range(w_start, w_end):
                    curr_idx = n * (channels * in_height * in_width) + c * (in_height * in_width) + h * in_width + w_pos
                    if curr_idx == max_idx:
                        continue
                    diff_a = a_max - A_in.flat[curr_idx]
                    diff_b = b_max - B_in.flat[curr_idx]
                    if abs(diff_b) <= SI_EPS:
                        if diff_a < -SI_EPS:
                            local_valid = 0
                    else:
                        threshold = -diff_a / diff_b
                        if diff_b > 0.0:
                            if threshold > local_lower:
                                local_lower = threshold
                        elif threshold < local_upper:
                            local_upper = threshold
        idx += cuda.gridsize(1)
    s_lower[tid] = local_lower
    s_upper[tid] = local_upper
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            if s_lower[tid + stride] > s_lower[tid]:
                s_lower[tid] = s_lower[tid + stride]
            if s_upper[tid + stride] < s_upper[tid]:
                s_upper[tid] = s_upper[tid + stride]
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        if local_valid == 0:
            cuda.atomic.min(valid_flag, 0, 0)
        if s_lower[0] > -1e+200:
            cuda.atomic.max(itv_lower, 0, s_lower[0])
        if s_upper[0] < 1e+200:
            cuda.atomic.min(itv_upper, 0, s_upper[0])

@cuda.jit
def linear_forward_kernel_naive(input_data, weight, bias, output, N, in_features, out_features, use_bias):
    idx = cuda.grid(1)
    total_elements = N * out_features
    while idx < total_elements:
        out_f = idx % out_features
        n = idx // out_features
        sum_val = 0.0
        for in_f in range(in_features):
            sum_val += input_data[n, in_f] * weight[out_f, in_f]
        if use_bias:
            sum_val += bias[out_f]
        output[n, out_f] = sum_val
        idx += cuda.gridsize(1)

@cuda.jit
def linear_forward_kernel_tiled(input_data, weight, bias, output, N, in_features, out_features, use_bias):
    tile_size = 16
    s_input = cuda.shared.array((16, 16), dtype=float32)
    s_weight = cuda.shared.array((16, 16), dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    n = by * tile_size + ty
    out_f = bx * tile_size + tx
    sum_val = 0.0
    num_tiles = (in_features + tile_size - 1) // tile_size
    for tile_idx in range(num_tiles):
        in_f = tile_idx * tile_size + tx
        if n < N and in_f < in_features:
            s_input[ty, tx] = input_data[n, in_f]
        else:
            s_input[ty, tx] = 0.0
        in_f_w = tile_idx * tile_size + ty
        if out_f < out_features and in_f_w < in_features:
            s_weight[tx, ty] = weight[out_f, in_f_w]
        else:
            s_weight[tx, ty] = 0.0
        cuda.syncthreads()
        for k in range(tile_size):
            sum_val += s_input[ty, k] * s_weight[tx, k]
        cuda.syncthreads()
    if n < N and out_f < out_features:
        if use_bias:
            sum_val += bias[out_f]
        output[n, out_f] = sum_val