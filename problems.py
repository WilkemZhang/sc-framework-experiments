"""Four synthetic full-batch objectives for testing the strongly convex theorem."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import pdist
from scipy.special import expit
from scipy.optimize import minimize

FAMILIES = ("ridge", "logistic", "asymmetric", "logsumexp")
KAPPAS = (2, 10, 100, 1000, 10000, 100000, 1000000)


def softmax(z):
    shifted = z - z.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def exp_remainder(h):
    """exp(h)-1-h; a Taylor polynomial for small h preserves the quadratic term."""
    if np.max(np.abs(h)) < 0.008:
        return h*h*(0.5 + h*(1/6 + h*(1/24 + h*(1/120 + h/720))))
    out = np.expm1(h) - h
    small = np.abs(h) < 1e-4
    s = h[small]
    out[small] = s*s*(0.5 + s*(1/6 + s/24))
    return out


def log_partition_bregman(prob, delta):
    # D_logsumexp(z+delta,z) = log(1 + E_p[exp(h)-1-h]), h=delta-E_p[delta].
    h = delta - np.sum(prob*delta, axis=-1, keepdims=True)
    if np.max(h) > 600:
        # Can occur in far-away line-search trials; at large h little absolute
        # precision is required. Ignore exactly underflowed probabilities.
        from scipy.special import logsumexp
        return logsumexp(h, b=prob, axis=-1)
    return np.log1p(np.sum(prob*exp_remainder(h), axis=-1))


class Problem:
    def __init__(self, family, replicate):
        # A fixed random stream reproduces the same instance across methods.
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
                self.sign = 2*rng.integers(0, 2, size=self.n) - 1
                self.C = np.linalg.norm(self.A, ord=2)**2 / (4*self.n)
            else:
                self.offset = rng.standard_normal(self.n)
                self.C = np.max(pdist(self.A, metric="sqeuclidean")) / 4
        self.x0 = rng.standard_normal(self.dim)

    def set_kappa(self, kappa):
        if self.family != "asymmetric":
            self.lam = self.C / (kappa - 1)
            self.mu = self.lam
            self.L = self.lam + self.C
        else:
            self.mu, self.L = 1.0, float(kappa)

    def oracle(self, x):
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
        if self.family == "ridge":
            return None
        if self.family == "logistic":
            z = self.A @ x
            return np.column_stack([expit(-z), expit(z)])
        return softmax(self.A @ x - self.offset)

    def bregman(self, x, base, prob=None):
        diff = x - base
        if self.family == "ridge":
            dz = self.A @ diff
            return 0.5*((dz @ dz)/self.n + self.lam*(diff @ diff))
        if self.family == "asymmetric":
            # The positive-part squared term has a simple, nonnegative divergence.
            pos_x = np.maximum(x - self.center, 0)
            pos_base = np.maximum(base - self.center, 0)
            change = pos_x - pos_base
            one_sided = 0.5*(change @ change) + pos_base @ np.maximum(self.center - x, 0)
            return 0.5*(diff @ diff) + (self.L-1)*one_sided

        if prob is None:
            prob = self.probabilities(base)
        dz = self.A @ diff
        if self.family == "logistic":
            dz = np.column_stack([np.zeros(self.n), dz])
        return 0.5*self.lam*(diff @ diff) + np.mean(log_partition_bregman(prob, dz))

    def hessian(self, x):
        """Used only to refine the independently computed reference point."""
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
        if self.family == "asymmetric":
            xstar = self.center.copy()
            gap0 = self.oracle(self.x0)[0]
            return xstar, np.zeros_like(xstar), None, gap0, 0.0
        if self.family == "ridge":
            gram = self.A @ self.A.T + (self.n*self.lam)*np.eye(self.n)
            x = self.A.T @ solve(gram, self.b, assume_a="pos")
        else:
            opt = minimize(self.oracle, np.zeros(self.dim), jac=True, method="L-BFGS-B",
                           options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12})
            x = opt.x
        for _ in range(12):
            _, grad = self.oracle(x)
            cert = (grad @ grad)/(2*self.mu)
            prob = self.probabilities(x)
            gap0 = self.bregman(self.x0, x, prob) + grad @ (self.x0 - x)
            d0 = np.linalg.norm(self.x0 - x)
            radius = np.linalg.norm(grad) / self.mu
            if gap0 > 0 and cert/gap0 <= 1e-20 and radius <= 1e-9*d0:
                return x, grad, prob, gap0, cert
            x = x - solve(self.hessian(x), grad, assume_a="pos")
        raise ArithmeticError("Reference missed its relative error certificate")
