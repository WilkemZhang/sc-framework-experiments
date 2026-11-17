"""Seven full-gradient methods used in the numerical experiments.

AC-FGM uses the separate two-step initialization: the accepted trial sets
eta_init, its secant sets L1, and tau_2 = 1.
See Li and Lan, Corollary 2, doi:10.1007/s10107-025-02250-z.
"""
import math

import numpy as np


METHODS = {
    "adanag_g12": "AdaNAG-G12",
    "adanag_g_half": "AdaNAG-G1/2",
    "ac_fgm_a01": "AC-FGM",
    "adaagm55": "AdaAGM5.5",
    "fg_original": "VC-FG",
    "widecap": "CC-FG",
    "nag_sc_oracle": "NAG-SC oracle",
}


def _adanag_coefficients(method, k):
    if method == "adanag_g12":
        return (k + 14) / 12, 0.5 * ((k + 3) / (k + 14)) ** 2
    return 2 * math.sqrt(k + 3), 0.5


def _coupled_rule(method, k):
    """theta_k, alpha_k, theta_(k+1), growth factor and curvature cap."""
    if method == "widecap":
        theta, next_theta = (k + 26) / 26, (k + 27) / 26
        growth = (k + 26) ** 2 / ((k + 27) * (k + 1))
        return theta, 7 / 5, next_theta, growth, 57 / 220
    if method == "fg_original":
        theta, next_theta = (k + 20) / 20, (k + 21) / 20
        alpha = 48 / 25 * ((k + 1) / (k + 20)) ** 2
        next_alpha = 48 / 25 * ((k + 2) / (k + 21)) ** 2
        growth = (k + 1) * (k + 21) / (k + 2) ** 2
        return theta, alpha, next_theta, growth, (2 - next_alpha) / 4

    prev_theta, prev_alpha = _adanag_coefficients(method, k - 1)
    theta, alpha = _adanag_coefficients(method, k)
    next_theta, next_alpha = _adanag_coefficients(method, k + 1)
    prev_a = 0.0 if k == 0 else alpha * theta * (theta - 1)
    a = next_alpha * next_theta * (next_theta - 1)
    b = (alpha * theta) ** 2 * ((theta - 1) ** 2 / (prev_alpha * prev_theta**2) - 1)
    next_b = (next_alpha * next_theta) ** 2 * ((next_theta - 1) ** 2 / (alpha * theta**2) - 1)
    growth = (prev_a + alpha * theta) / a
    cap = 1 / (a / b + (next_b + (next_alpha * next_theta) ** 2) / a)
    return theta, alpha, next_theta, growth, cap


def _local_curvature(bregman, x, g, x_next, g_next, previous):
    """Equation (11) uses D_f(x, x_next), with gradient at x_next."""
    difference = g_next - g
    numerator = float(np.vdot(difference, difference))
    if numerator == 0 and np.array_equal(x_next, x):
        return previous
    divergence = float(bregman(x, x_next))
    if not (math.isfinite(divergence) and divergence > 0):
        return math.nan
    return 0.5 * numerator / divergence


def run_solver(method, oracle, x0, mu, L, n_steps, *, bregman, monitor):
    """Run up to n_steps accepted iterations from x0 (k=0).

    monitor(k, x) inspects every accepted iterate, starting at k=0.
    A true return value ends the run. Only NAG uses mu and L.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    if n_steps < 0:
        raise ValueError("The iteration budget must be nonnegative.")
    x = np.array(x0, dtype=float, copy=True)
    k = 0

    def finish(status):
        return {"status": status, "iterations": k, "x": x}

    f, g = oracle(x)
    grad0_norm = float(np.linalg.norm(g))
    if not (math.isfinite(f) and math.isfinite(grad0_norm)):
        return finish("nonfinite_oracle")
    if grad0_norm == 0:
        monitor(k, x)
        return finish("stationary")

    if method == "nag_sc_oracle":
        if not 0 < mu <= L or not math.isfinite(L):
            raise ValueError("NAG requires finite 0 < mu <= L.")
        step = 1 / L
        beta = (math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))
        y = x
    else:
        probe = x - g / grad0_norm
        _, g_probe = oracle(probe)
        local = float(np.linalg.norm(g_probe - g) / np.linalg.norm(probe - x))
        if not (math.isfinite(local) and local > 0):
            return finish("invalid_initial_secant")

    if monitor(k, x):
        return finish("hit")
    if n_steps == 0:
        return finish("max_iter")

    if method == "ac_fgm_a01":
        # Only the accepted trial is an iterate.
        z = x.copy()
        beta = 1 - math.sqrt(6) / 3
        for trial in range(200):
            trial_lipschitz = local * 1.5**trial / 4
            eta_init = 2 / (5 * trial_lipschitz)
            x1 = x - eta_init * g
            f1, g1 = oracle(x1)
            difference_norm = float(np.linalg.norm(g1 - g))
            if math.isfinite(f1) and difference_norm <= 0.4 * grad0_norm:
                break
        else:
            return finish("initial_line_search_failed")
        local = difference_norm / np.linalg.norm(x1 - x)
        if not (math.isfinite(local) and local > 0):
            return finish("invalid_initial_secant")
        step = min((1 - beta) * eta_init, 1 / (4 * local))
        tau_old, tau = 0.0, 1.0
        k, x, f, g = 1, x1, f1, g1
        if monitor(k, x):
            return finish("hit")
    elif method == "adaagm55":
        theta, y = 2.0, x
        step = 1 / (12 * local)
    elif method != "nag_sc_oracle":
        z = x
        initial_multiplier = {
            "fg_original": 1 / 32, "widecap": 1 / 32,
            "adanag_g12": 21 / 802, "adanag_g_half": 2 * math.sqrt(3) / 5,
        }[method]
        step = initial_multiplier / local

    while k < n_steps:
        if method == "nag_sc_oracle":
            y_next = x - step * g
            x_next = y_next + beta * (y_next - y)
            y = y_next
        elif method == "ac_fgm_a01":
            if k >= 2:
                step = min(4 * step / 3, (1 + tau_old) * step / tau, tau / (4 * local))
                next_tau = tau + 0.1 / 2 + 2 * (1 - 0.1) * step * local / tau
                tau_old, tau = tau, next_tau
            # k=1 uses tau_2=1 and the separately initialized eta_2.
            y_ac = z - step * g
            z = z - beta * step * g
            x_next = (tau * x + y_ac) / (1 + tau)
        elif method == "adaagm55":
            next_theta = (1 / 8 + math.sqrt((1 / 8) ** 2 + 4 * theta**2)) / 2
            y_next = x - step * g
            x_next = (y_next + (theta - 1) / next_theta * (y_next - y)
                      - 0.5 * theta / next_theta * (y_next - x))
            y, theta = y_next, next_theta
            growth = min((theta - 1 / 8) / (theta - 1), 2 * (1 - 1 / theta))
            cap = 1 / (2 * theta / (theta - 1) + 8 * (theta - 1) / (theta - 1 / 8))
        else:
            theta, alpha, next_theta, growth, cap = _coupled_rule(method, k)
            y = x - step * g
            z = z - step * alpha * theta * g
            x_next = (1 - 1 / next_theta) * y + z / next_theta

        if not np.isfinite(x_next).all():
            return finish("nonfinite_iterate")
        f_next, g_next = oracle(x_next)
        if not math.isfinite(f_next) or not np.isfinite(g_next).all():
            return finish("nonfinite_oracle")
        previous_x, previous_g = x, g
        k, x, f, g = k + 1, x_next, f_next, g_next
        if monitor(k, x):
            return finish("hit")
        if method != "nag_sc_oracle" and k < n_steps:
            local = _local_curvature(bregman, previous_x, previous_g, x, g, local)
            if not (math.isfinite(local) and local > 0):
                return finish("invalid_local_curvature")
            if method != "ac_fgm_a01":
                step = min(growth * step, cap / local)
    return finish("max_iter")
