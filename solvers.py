"""The seven gradient methods used in the experiments."""

import math

import numpy as np


METHODS = {
    "adanag_g12": (r"AdaNAG-G$_{12}$", "#6a3d9a", "--"),
    "adanag_g_half": (r"AdaNAG-G$_{1/2}$", "#b8860b", "--"),
    "ac_fgm_a01": ("AC-FGM", "#008b45", "--"),
    "adaagm55": (r"AdaAGM$_{5.5}$", "#c51b7d", "--"),
    "fg_original": ("VC-FG", "#0066ff", "-"),
    "widecap": ("CC-FG", "#e31a1c", "-"),
    "nag_sc_oracle": ("NAG-SC (oracle)", "#444444", "--"),
}


# Seven gradient methods: initialization, one update, and the next stepsize.
# Only the oracle NAG-SC baseline uses the global constants mu and L.


def _adanag_coefficients(method, k):
    if method == "adanag_g12":
        return (k + 14) / 12, 0.5 * ((k + 3) / (k + 14)) ** 2
    return 2 * math.sqrt(k + 3), 0.5


def _coupled_rule(method, k):
    """Return theta_k, alpha_k, theta_(k+1), growth limit, and curvature cap."""
    if method == "widecap":                         # CC-FG
        theta, next_theta = (k + 26) / 26, (k + 27) / 26
        growth = (k + 26) ** 2 / ((k + 27) * (k + 1))
        return theta, 7 / 5, next_theta, growth, 57 / 220
    if method == "fg_original":                     # VC-FG
        theta, next_theta = (k + 20) / 20, (k + 21) / 20
        alpha = 48 / 25 * ((k + 1) / (k + 20)) ** 2
        next_alpha = 48 / 25 * ((k + 2) / (k + 21)) ** 2
        growth = (k + 1) * (k + 21) / (k + 2) ** 2
        return theta, alpha, next_theta, growth, (2 - next_alpha) / 4

    # AdaNAG-G: coefficients A_(k-1), A_k, B_k, and B_(k+1).
    prev_theta, prev_alpha = _adanag_coefficients(method, k - 1)
    theta, alpha = _adanag_coefficients(method, k)
    next_theta, next_alpha = _adanag_coefficients(method, k + 1)
    prev_a = 0.0 if k == 0 else alpha * theta * (theta - 1)
    a = next_alpha * next_theta * (next_theta - 1)
    # Extending the formula to k=-1 specifies B_0; A_(-1) remains zero.
    b = (alpha * theta) ** 2 * ((theta - 1) ** 2 / (prev_alpha * prev_theta**2) - 1)
    next_b = (next_alpha * next_theta) ** 2 * ((next_theta - 1) ** 2 / (alpha * theta**2) - 1)
    growth = (prev_a + alpha * theta) / a
    cap = 1 / (a / b + (next_b + (next_alpha * next_theta) ** 2) / a)
    return theta, alpha, next_theta, growth, cap


def _local_curvature(bregman, x, g, x_next, g_next, previous):
    """L_k = ||g_k-g_(k-1)||^2 / [2 D_f(x_(k-1), x_k)]."""
    difference = g_next - g
    numerator = float(np.vdot(difference, difference))
    if numerator == 0 and np.array_equal(x_next, x):
        return previous
    return 0.5 * numerator / float(bregman(x, x_next))


def run_solver(method, oracle, x0, mu, L, n_steps, *, bregman, monitor):
    """Apply one method until both errors meet the target or n_steps is reached.

    oracle(x) returns f(x) and its gradient. monitor(k, x) records the errors
    at each accepted iterate and returns True when the final target is reached.
    """
    x = np.array(x0, dtype=float, copy=True)
    k = 0

    def finish(status):
        return {"status": status, "iterations": k, "x": x}

    f, g = oracle(x)
    grad0_norm = float(np.linalg.norm(g))
    if grad0_norm == 0:
        monitor(k, x)
        return finish("stationary")
    if monitor(k, x):
        return finish("hit")

    if method == "nag_sc_oracle":
        step = 1 / L
        beta = (math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))
        y = x
    else:
        # A unit-length gradient displacement gives the initial secant estimate.
        probe = x - g / grad0_norm
        _, g_probe = oracle(probe)
        local = float(np.linalg.norm(g_probe - g) / np.linalg.norm(probe - x))

    if method == "ac_fgm_a01":
        # Rejected initialization trials do not count as accepted iterates.
        z = x.copy()
        beta = 1 - math.sqrt(6) / 3
        trial = 0
        while True:
            trial_lipschitz = local * 1.5**trial / 4
            eta_init = 2 / (5 * trial_lipschitz)
            x1 = x - eta_init * g
            f1, g1 = oracle(x1)
            difference_norm = float(np.linalg.norm(g1 - g))
            if difference_norm <= 0.4 * grad0_norm:
                break
            trial += 1
        local = difference_norm / np.linalg.norm(x1 - x)
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
            # The second update uses the separately initialized eta_2 and tau_2=1.
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

        f_next, g_next = oracle(x_next)
        previous_x, previous_g = x, g
        k, x, f, g = k + 1, x_next, f_next, g_next
        if monitor(k, x):
            return finish("hit")
        if method != "nag_sc_oracle" and k < n_steps:
            local = _local_curvature(bregman, previous_x, previous_g, x, g, local)
            if method != "ac_fgm_a01":
                step = min(growth * step, cap / local)
    return finish("max_iter")
