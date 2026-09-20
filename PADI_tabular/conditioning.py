import numpy as np
from ..util import (
    solve_linear_inequality_geq,
    intersect,
    solve_linear_inequality,
    solve_quadratic_inequality_batch,
    intersect_intervals as intersect_intervals_tuples,
    get_test_statistic_interval_vs_mean
)


def get_final_interval(X, model, a, b, R_squared, center_c, alpha=0.01, zk=None, force_rebuild_cache=False):
    from .si_padi import propagate_xab
    sub_itv, A_final, B_final, X_final = propagate_xab(X, a, b, model, zk=zk, force_rebuild_cache=force_rebuild_cache)
    if sub_itv is None:
        return (None, None)
    network_interval = [tuple(sub_itv)]
    center_c = np.asarray(center_c, dtype=np.float64).reshape(-1)
    score = float(np.sum((X_final - center_c) ** 2))
    Oz = {0} if score > float(R_squared) else set()
    diff = A_final - center_c
    u = float(np.sum(diff ** 2) - float(R_squared))
    v = float(2.0 * np.sum(diff * B_final))
    w = float(np.sum(B_final ** 2))
    if len(Oz) == 0:
        anomaly_intervals = solve_quadratic_inequality_batch(np.array([u], dtype=np.float64), np.array([v], dtype=np.float64), np.array([w], dtype=np.float64))
    else:
        anomaly_intervals = solve_quadratic_inequality_batch(np.array([-u], dtype=np.float64), np.array([-v], dtype=np.float64), np.array([-w], dtype=np.float64))
    if anomaly_intervals is None or len(anomaly_intervals) == 0 or len(anomaly_intervals[0]) == 0:
        return (None, Oz)
    final_intervals = intersect_intervals_tuples([network_interval, anomaly_intervals[0]])
    if final_intervals is None or len(final_intervals) == 0:
        return (None, Oz)
    return (final_intervals, Oz)