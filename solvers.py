"""Paper algorithms, with one update loop per method.

Read run_solver first, then the selected method for its coefficients,
iterate updates, and stepsize rule.

Structural reference: Li and Lan's AC-FGM implementation separates the problem
oracle from individual solver routines:
https://github.com/tli432/AC-FGM-Implementation
The update formulas and initialization policies here follow our manuscript.
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


def _local_curvature(bregman, x_k, g_k, x_next, g_next, L_k):
    """Equation (11) uses D_f(x_k, x_next), with gradient at x_next."""
    difference = g_next - g_k
    numerator = float(np.vdot(difference, difference))
    # At a repeated point with the same gradient the ratio is 0/0.
    if numerator == 0 and np.array_equal(x_next, x_k):
        return L_k
    return 0.5 * numerator / float(bregman(x_k, x_next))


def _result(status, k, x):
    """Return the last accepted iterate and its iteration index."""
    return {"status": status, "iterations": k, "x": x}


def run_solver(method, oracle, x0, mu, L, n_steps, *, bregman, monitor):
    """Initialize once, record x0, and call the requested algorithm.

    oracle(x) returns (f(x), gradient). bregman(x, base) evaluates D_f(x, base).
    monitor(k, x) records an accepted iterate and returns True to stop.
    n_steps counts accepted updates, not oracle calls. Only NAG uses mu and L.
    """
    x = np.array(x0, dtype=float, copy=True)
    _, gradient = oracle(x)
    grad0_norm = float(np.linalg.norm(gradient))
    if grad0_norm == 0:
        monitor(0, x)
        return _result("stationary", 0, x)

    if method != "nag_sc_oracle":
        # The paper uses this same initial secant for all adaptive methods.
        probe = x - gradient / grad0_norm
        _, probe_gradient = oracle(probe)
        local = float(np.linalg.norm(probe_gradient - gradient) / np.linalg.norm(probe - x))

    if monitor(0, x):
        return _result("hit", 0, x)

    if method == "nag_sc_oracle":
        return run_nag_sc(oracle, x, gradient, mu, L, n_steps, monitor)
    adaptive_methods = {
        "fg_original": run_vc_fg,
        "widecap": run_cc_fg,
        "adanag_g12": run_adanag_g12,
        "adanag_g_half": run_adanag_g_half,
        "ac_fgm_a01": run_ac_fgm,
        "adaagm55": run_adaagm,
    }
    solve = adaptive_methods[method]
    return solve(oracle, x, gradient, local, n_steps, bregman, monitor)


def run_vc_fg(oracle, x_k, g_k, L_k, n_steps, bregman, monitor):
    """Run VC-FG from its initial stepsize through the accepted updates."""
    p, alpha_bar = 20, 48 / 25
    z_k = x_k
    s_k = (1 / 32) / L_k

    for k in range(n_steps):
        theta_k = (k + p) / p
        theta_next = (k + p + 1) / p
        alpha_k = alpha_bar * ((k + 1) / (k + p)) ** 2
        alpha_next = alpha_bar * ((k + 2) / (k + p + 1)) ** 2
        U_next = (k + 1) * (k + p + 1) / (k + 2) ** 2
        q_next = (2 - alpha_next) / 4

        y_next = x_k - s_k * g_k
        z_next = z_k - s_k * alpha_k * theta_k * g_k
        x_next = (1 - 1 / theta_next) * y_next + z_next / theta_next

        _, g_next = oracle(x_next)
        if monitor(k + 1, x_next):
            return _result("hit", k + 1, x_next)
        if k + 1 < n_steps:
            L_next = _local_curvature(bregman, x_k, g_k, x_next, g_next, L_k)
            s_k = min(U_next * s_k, q_next / L_next)
            L_k = L_next
        x_k, z_k, g_k = x_next, z_next, g_next
    return _result("max_iter", n_steps, x_k)


def run_cc_fg(oracle, x_k, g_k, L_k, n_steps, bregman, monitor):
    """Run CC-FG from its initial stepsize through the accepted updates."""
    p, alpha_k = 26, 7 / 5
    q_next = 57 / 220
    z_k = x_k
    s_k = (1 / 32) / L_k

    for k in range(n_steps):
        theta_k = (k + p) / p
        theta_next = (k + p + 1) / p
        U_next = (k + p) ** 2 / ((k + p + 1) * (k + 1))

        y_next = x_k - s_k * g_k
        z_next = z_k - s_k * alpha_k * theta_k * g_k
        x_next = (1 - 1 / theta_next) * y_next + z_next / theta_next

        _, g_next = oracle(x_next)
        if monitor(k + 1, x_next):
            return _result("hit", k + 1, x_next)
        if k + 1 < n_steps:
            L_next = _local_curvature(bregman, x_k, g_k, x_next, g_next, L_k)
            s_k = min(U_next * s_k, q_next / L_next)
            L_k = L_next
        x_k, z_k, g_k = x_next, z_next, g_next
    return _result("max_iter", n_steps, x_k)


def run_adanag_g12(oracle, x_k, g_k, L_k, n_steps, bregman, monitor):
    """Run AdaNAG-G12 from its initial stepsize through the accepted updates."""
    z_k = x_k
    s_k = (21 / 802) / L_k

    for k in range(n_steps):
        theta_prev, theta_k, theta_next = (k + 13) / 12, (k + 14) / 12, (k + 15) / 12
        alpha_prev = 0.5 * ((k + 2) / (k + 13)) ** 2
        alpha_k = 0.5 * ((k + 3) / (k + 14)) ** 2
        alpha_next = 0.5 * ((k + 4) / (k + 15)) ** 2
        A_G_prev = 0.0 if k == 0 else alpha_k * theta_k * (theta_k - 1)
        A_G_k = alpha_next * theta_next * (theta_next - 1)
        B_G_k = (alpha_k * theta_k) ** 2 * (
            (theta_k - 1) ** 2 / (alpha_prev * theta_prev**2) - 1
        )
        B_G_next = (alpha_next * theta_next) ** 2 * (
            (theta_next - 1) ** 2 / (alpha_k * theta_k**2) - 1
        )
        s_first_bound_next = ((A_G_prev + alpha_k * theta_k) / A_G_k) * s_k
        q_next = 1 / (A_G_k / B_G_k + (B_G_next + (alpha_next * theta_next) ** 2) / A_G_k)

        y_next = x_k - s_k * g_k
        z_next = z_k - s_k * alpha_k * theta_k * g_k
        x_next = (1 - 1 / theta_next) * y_next + z_next / theta_next

        _, g_next = oracle(x_next)
        if monitor(k + 1, x_next):
            return _result("hit", k + 1, x_next)
        if k + 1 < n_steps:
            L_next = _local_curvature(bregman, x_k, g_k, x_next, g_next, L_k)
            s_k = min(s_first_bound_next, q_next / L_next)
            L_k = L_next
        x_k, z_k, g_k = x_next, z_next, g_next
    return _result("max_iter", n_steps, x_k)


def run_adanag_g_half(oracle, x_k, g_k, L_k, n_steps, bregman, monitor):
    """Run AdaNAG-G1/2 from its initial stepsize through the accepted updates."""
    z_k = x_k
    s_k = (2 * math.sqrt(3) / 5) / L_k

    for k in range(n_steps):
        theta_prev = 2 * math.sqrt(k + 2)
        theta_k = 2 * math.sqrt(k + 3)
        theta_next = 2 * math.sqrt(k + 4)
        alpha_prev = alpha_k = alpha_next = 0.5
        A_G_prev = 0.0 if k == 0 else alpha_k * theta_k * (theta_k - 1)
        A_G_k = alpha_next * theta_next * (theta_next - 1)
        B_G_k = (alpha_k * theta_k) ** 2 * (
            (theta_k - 1) ** 2 / (alpha_prev * theta_prev**2) - 1
        )
        B_G_next = (alpha_next * theta_next) ** 2 * (
            (theta_next - 1) ** 2 / (alpha_k * theta_k**2) - 1
        )
        s_first_bound_next = ((A_G_prev + alpha_k * theta_k) / A_G_k) * s_k
        q_next = 1 / (A_G_k / B_G_k + (B_G_next + (alpha_next * theta_next) ** 2) / A_G_k)

        y_next = x_k - s_k * g_k
        z_next = z_k - s_k * alpha_k * theta_k * g_k
        x_next = (1 - 1 / theta_next) * y_next + z_next / theta_next

        _, g_next = oracle(x_next)
        if monitor(k + 1, x_next):
            return _result("hit", k + 1, x_next)
        if k + 1 < n_steps:
            L_next = _local_curvature(bregman, x_k, g_k, x_next, g_next, L_k)
            s_k = min(s_first_bound_next, q_next / L_next)
            L_k = L_next
        x_k, z_k, g_k = x_next, z_next, g_next
    return _result("max_iter", n_steps, x_k)


def run_ac_fgm(oracle, x, gradient, local, n_steps, bregman, monitor):
    """Li and Lan, Corollary 2: first-step search, second step, then recurrence.

    x, gradient, and local have already been initialized by run_solver.
    z is the prox-center (called y in Li and Lan); y_ac is their trial z.
    The second accepted update uses tau_2=1 without advancing it early.
    """
    # The first accepted AC-FGM step is outside the loop.
    if n_steps == 0:
        return _result("max_iter", 0, x)
    z = x.copy()
    beta = 1 - math.sqrt(6) / 3
    grad0_norm = float(np.linalg.norm(gradient))
    trial = 0
    while True:
        trial_lipschitz = local * 1.5**trial / 4
        eta_init = 2 / (5 * trial_lipschitz)
        x1 = x - eta_init * gradient
        _, gradient1 = oracle(x1)
        difference_norm = float(np.linalg.norm(gradient1 - gradient))
        if difference_norm <= 0.4 * grad0_norm:
            break
        trial += 1

    # Use the measured first-step secant, not the line-search trial constant.
    local = difference_norm / np.linalg.norm(x1 - x)
    step = min((1 - beta) * eta_init, 1 / (4 * local))
    tau_prev, tau = 0.0, 1.0
    x, gradient = x1, gradient1
    if monitor(1, x):
        return _result("hit", 1, x)

    for k in range(1, n_steps):
        if k >= 2:
            step = min(4 * step / 3, (1 + tau_prev) * step / tau, tau / (4 * local))
            tau_next = tau + 0.1 / 2 + 2 * (1 - 0.1) * step * local / tau
            tau_prev, tau = tau, tau_next
        y_ac = z - step * gradient
        z = z - beta * step * gradient
        x_next = (tau * x + y_ac) / (1 + tau)

        _, g_next = oracle(x_next)
        if monitor(k + 1, x_next):
            return _result("hit", k + 1, x_next)
        if k + 1 < n_steps:
            local = _local_curvature(bregman, x, gradient, x_next, g_next, local)
        x, gradient = x_next, g_next
    return _result("max_iter", n_steps, x)


def run_adaagm(oracle, x, gradient, local, n_steps, bregman, monitor):
    """AdaAGM Corollary 5.5 with (m, gamma, beta, theta_0)=(1/8, 1/2, 1, 2).

    The curvature rule corresponds to omega=eta=1/2 in the general notation.

    x, gradient, and local have already been initialized by run_solver.
    Update theta and the extrapolated point, then choose the next stepsize.
    """
    theta, y = 2.0, x
    step = 1 / (12 * local)
    for k in range(n_steps):
        theta_next = (1 / 8 + math.sqrt((1 / 8) ** 2 + 4 * theta**2)) / 2
        y_next = x - step * gradient
        x_next = (y_next + (theta - 1) / theta_next * (y_next - y)
                  - 0.5 * theta / theta_next * (y_next - x))
        y, theta = y_next, theta_next
        growth = min((theta - 1 / 8) / (theta - 1), 2 * (1 - 1 / theta))
        cap = 1 / (2 * theta / (theta - 1) + 8 * (theta - 1) / (theta - 1 / 8))

        _, g_next = oracle(x_next)
        if monitor(k + 1, x_next):
            return _result("hit", k + 1, x_next)
        if k + 1 < n_steps:
            # AdaAGM's native rule uses Lhat=0 for equal gradients, so the
            # curvature cap is infinite and only the growth bound remains.
            if np.array_equal(g_next, gradient):
                step = growth * step
            else:
                local = _local_curvature(bregman, x, gradient, x_next, g_next, local)
                step = min(growth * step, cap / local)
        x, gradient = x_next, g_next
    return _result("max_iter", n_steps, x)


def run_nag_sc(oracle, x, gradient, mu, L, n_steps, monitor):
    """NAG-SC with global bounds; x is the extrapolated, gradient-query point.

    x and gradient have already been initialized by run_solver.
    """
    step = 1 / L
    beta = (math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))
    y = x
    for k in range(n_steps):
        y_next = x - step * gradient
        x_next = y_next + beta * (y_next - y)
        y = y_next

        _, g_next = oracle(x_next)
        if monitor(k + 1, x_next):
            return _result("hit", k + 1, x_next)
        x, gradient = x_next, g_next
    return _result("max_iter", n_steps, x)
