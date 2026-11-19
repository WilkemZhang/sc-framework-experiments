"""Problem families, experiment parameters, and reference solutions."""

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.special import expit, logsumexp


NUMBER_OF_DRAWS = 5
KAPPAS = (2, 10, 100, 1000, 10000, 100000, 1000000)
EXPONENTS = (8, 12, 15)                # Targets: 1e-8, 1e-12, 1e-15
FAMILIES = ("ridge", "logistic", "asymmetric", "logsumexp")  # Seed order


# Stable loss differences are needed when the target error is as small as 1e-15.
# These formulas avoid subtracting two almost equal objective values.
def softmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def exp_remainder(h):
    """Evaluate exp(h)-1-h accurately, including when h is very small."""
    h = np.asarray(h)
    if np.max(np.abs(h)) < 0.008:
        return h*h*(0.5 + h*(1/6 + h*(1/24 + h*(1/120 + h/720))))
    out = np.array(np.expm1(h) - h)
    small = np.abs(h) < 1e-4
    s = h[small]
    out[small] = s*s*(0.5 + s*(1/6 + s/24))
    return out


def log_partition_bregman(prob, delta, log_prob=None):
    """Stable divergence for the logistic and log-sum-exp losses."""
    h = delta - np.sum(prob*delta, axis=-1, keepdims=True)
    if np.max(h) > 600 or log_prob is not None:
        # Log probabilities retain contributions too small for ordinary probabilities.
        if log_prob is None:
            with np.errstate(divide="ignore"):
                log_prob = np.log(prob)
        log_rem = np.full_like(h, -np.inf, dtype=float)
        large = h > 50
        t = h[large]
        log_rem[large] = t + np.log1p(-(1+t)*np.exp(-t))
        remainder = exp_remainder(h[~large]) if np.any(~large) else np.empty(0)
        with np.errstate(divide="ignore"):
            log_rem[~large] = np.log(remainder)
        return np.logaddexp(0, logsumexp(log_prob + log_rem, axis=-1))
    return np.log1p(np.sum(prob*exp_remainder(h), axis=-1))


class Problem:
    """One sampled problem, shared by every method and conditioning level."""

    def __init__(self, family, replicate):
        rng = np.random.default_rng(np.random.SeedSequence([replicate, FAMILIES.index(family)]))
        self.family = family
        self.dim = self.p = 32
        self.n = self.dim
        self.C = None
        if family == "asymmetric":
            self.center = rng.standard_normal(self.dim)
        elif family == "ridge":
            self.n = 24
            self.A = rng.uniform(0, 1, size=(self.n, self.dim))
            direction = rng.standard_normal(self.dim)
            radius = rng.random()**(1/self.dim)
            u = radius * direction / np.linalg.norm(direction)
            self.b = self.A @ u
            self.C = np.linalg.norm(self.A, ord=2)**2 / self.n
        else:
            self.n = 128
            self.A = rng.standard_normal((self.n, self.dim))
            if family == "logistic":
                self.sign = rng.choice([-1, 1], size=self.n)
                self.C = np.linalg.norm(self.A, ord=2)**2 / (4*self.n)
            else:
                self.offset = rng.standard_normal(self.n)
                self.C = np.max(pdist(self.A, metric="sqeuclidean")) / 4
        self.x0 = rng.standard_normal(self.dim)

    def set_kappa(self, kappa):
        # C bounds the unregularized curvature; lambda makes L/mu = kappa.
        kappa = float(kappa)
        if self.family == "asymmetric":
            self.mu, self.L = 1.0, kappa
        else:
            self.lam = self.C / (kappa - 1)
            self.mu = self.lam
            self.L = self.lam + self.C

    def oracle(self, x):
        """Return the objective value f(x) and gradient g(x)."""
        if self.family == "asymmetric":
            residual = x - self.center
            g = np.where(residual > 0, self.L*residual, residual)
            return 0.5*(residual @ g), g
        if self.family == "ridge":
            residual = self.A @ x - self.b
            value = 0.5*(residual @ residual)/self.n
            g = self.A.T @ residual / self.n
        elif self.family == "logistic":
            z = -self.sign*(self.A @ x)
            value = np.mean(np.logaddexp(0, z))
            g = self.A.T @ (-self.sign*expit(z)) / self.n
        else:
            z = self.A @ x - self.offset
            top = z.max()
            ex = np.exp(z-top)
            total = ex.sum()
            value = top + np.log(total)
            g = self.A.T @ (ex/total)
        return value + 0.5*self.lam*(x @ x), g + self.lam*x

    def probabilities(self, x):
        if self.family == "logistic":
            z = self.A @ x
            return np.column_stack([expit(-z), expit(z)])
        return softmax(self.A @ x - self.offset)

    def bregman(self, x, base):
        """Compute D_f(x, base) = f(x) - f(base) - grad f(base) @ (x - base)."""
        diff = x - base

        if self.family == "ridge":
            dz = self.A @ diff
            return 0.5 * ((dz @ dz) / self.n + self.lam * (diff @ diff))

        if self.family == "asymmetric":
            pos_x = np.maximum(x - self.center, 0)
            pos_base = np.maximum(base - self.center, 0)
            change = pos_x - pos_base
            one_sided = 0.5 * (change @ change) + pos_base @ np.maximum(self.center - x, 0)
            return 0.5 * (diff @ diff) + (self.L - 1) * one_sided

        prob = self.probabilities(base)
        dz = self.A @ diff
        if self.family == "logistic":
            dz = np.column_stack([np.zeros(self.n), dz])

        log_prob = None
        if np.any(prob < np.finfo(float).tiny):
            z = self.A @ base
            if self.family == "logistic":
                log_prob = np.column_stack([-np.logaddexp(0, z), -np.logaddexp(0, -z)])
            else:
                shifted = z - self.offset
                shifted = shifted - shifted.max()
                log_prob = shifted - np.log(np.exp(shifted).sum())

        return 0.5 * self.lam * (diff @ diff) + np.mean(log_partition_bregman(prob, dz, log_prob))

    def reference_hessian(self, x):
        """Used only to refine the reference minimizer, never by a compared method."""
        prob = self.probabilities(x)
        if self.family == "ridge":
            H = self.A.T @ self.A / self.n
        elif self.family == "logistic":
            H = (self.A.T * np.prod(prob, axis=1)) @ self.A / self.n
        else:
            centered = self.A - prob @ self.A
            H = (centered.T*prob) @ centered
        H.flat[::self.dim+1] += self.lam
        return H

    def reference(self):
        """Find an accurate reference minimizer, its gradient, and the initial gap."""
        if self.family == "asymmetric":
            xstar = self.center.copy()
            return xstar, np.zeros_like(xstar), self.oracle(self.x0)[0]
        if self.family == "ridge":
            gram = self.A @ self.A.T + (self.n*self.lam)*np.eye(self.n)
            x = self.A.T @ solve(gram, self.b, assume_a="pos")
        else:
            fit = minimize(self.oracle, np.zeros(self.dim), jac=True, method="L-BFGS-B",
                           options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12})
            x = fit.x
        # Newton refinement makes the reference error smaller than the plotted errors.
        for _ in range(12):
            _, grad = self.oracle(x)
            cert = (grad @ grad)/(2*self.mu)
            gap0 = self.bregman(self.x0, x) + grad @ (self.x0 - x)
            d0 = np.linalg.norm(self.x0 - x)
            radius = np.linalg.norm(grad) / self.mu
            if gap0 > 0 and cert/gap0 <= 1e-20 and radius <= 1e-9*d0:
                return x, grad, gap0
            x = x - solve(self.reference_hessian(x), grad, assume_a="pos")
        raise ArithmeticError("The reference solution needs more accuracy for a 1e-15 experiment.")
