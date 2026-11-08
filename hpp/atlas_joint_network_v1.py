"""Small genuinely shared multi-task network; numerical core only.

No data access, cohort construction, scaling, privacy export or model selection.
Caller fits every transformation on training data and supplies task-specific Z
without an intercept. Never put a realised follow-up interval in the binary head.
Model arrays and optimisation metadata are internal, not approved export records.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

VERSION = "atlas_joint_network_v1.0"


def _validated(X, Z, Y, mask, kinds, weights):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    kinds = tuple(kinds)
    Z = [np.asarray(z, dtype=float) for z in Z]
    if X.ndim != 2 or Y.ndim != 2 or mask.shape != Y.shape:
        raise ValueError("X/Y must be matrices; mask must have Y shape")
    n, t = Y.shape
    if len(X) != n or len(Z) != t or len(kinds) != t:
        raise ValueError("inconsistent sample/task dimensions")
    if any(z.ndim != 2 or len(z) != n for z in Z):
        raise ValueError("each Z must be a matrix with the same sample count")
    if not np.isfinite(X).all() or any(not np.isfinite(z).all() for z in Z):
        raise ValueError("caller must impute and scale X and Z before fitting")
    if not np.isfinite(Y[mask]).all():
        raise ValueError("observed targets must be finite")
    if not all(k in ("continuous", "binary") for k in kinds):
        raise ValueError("kinds must be continuous or binary")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be a finite nonnegative sample vector")
    if not np.any(w > 0):
        raise ValueError("positive training weight required")
    for j, k in enumerate(kinds):
        if k == "binary" and not np.isin(Y[mask[:, j], j], [0., 1.]).all():
            raise ValueError("observed binary targets must be 0 or 1")
    return X, Z, Y, mask, kinds, w


class Objective:
    """Prepared analytic objective; loss_gradient accepts a flat parameter vector.

    H=tanh(XW+b), eta=H V+c + each Z_j a_j. The average over active
    tasks contains weighted MSE or CE/training-null-entropy. L2/2 penalises
    ||W||^2 + (||V||^2 + sum_j ||a_j||^2)/n_active; b and c are unpenalised.
    Averaging head penalties matches each head's independent-task penalty scale.
    No target mask renormalisation
    depends on predictions. A task with no positive observed weight is inactive.
    """

    def __init__(self, X, Z, Y, mask, kinds, weights=None, hidden=4, l2=.01):
        self.X, self.Z, self.Y, self.mask, self.kinds, self.w = _validated(
            X, Z, Y, mask, kinds, weights)
        if int(hidden) != hidden or hidden < 0 or not np.isfinite(l2) or l2 < 0:
            raise ValueError("hidden must be nonnegative integer and l2 nonnegative")
        self.hidden, self.l2 = int(hidden), float(l2)
        self.n, self.p = self.X.shape
        self.t = self.Y.shape[1]
        self.q = [z.shape[1] for z in self.Z]
        shapes = [(self.p, self.hidden), (self.hidden,),
                  (self.hidden, self.t), (self.t,)] + [(q,) for q in self.q]
        self.shapes = shapes
        self.slices, at = [], 0
        for shape in shapes:
            size = int(np.prod(shape))
            self.slices.append(slice(at, at + size))
            at += size
        self.size = at
        self.indices, self.norm, self.null_entropy = [], [], []
        self.target_mean = []
        for j, kind in enumerate(self.kinds):
            ix = np.flatnonzero(self.mask[:, j] & (self.w > 0))
            self.indices.append(ix)
            total = float(self.w[ix].sum())
            self.norm.append(total)
            mean = float(np.average(self.Y[ix, j], weights=self.w[ix])) if total else 0.
            self.target_mean.append(mean)
            if kind == "binary" and total:
                if not 0 < mean < 1:
                    raise ValueError("active binary task requires both training classes")
                entropy = -mean*np.log(mean) - (1-mean)*np.log1p(-mean)
            else:
                entropy = 1.
            self.null_entropy.append(entropy)
        self.active = np.asarray(self.norm) > 0
        self.nactive = int(self.active.sum())
        if not self.nactive:
            raise ValueError("no task has positive observed training weight")

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.size,):
            raise ValueError("wrong parameter vector size")
        return [theta[s].reshape(shape) for s, shape in zip(self.slices, self.shapes)]

    def initial(self, seed=17):
        rng = np.random.default_rng(seed)
        parts = [np.zeros(shape) for shape in self.shapes]
        W, b, V, c, *a = parts
        if self.hidden:
            W[:] = rng.normal(0, .2/np.sqrt(max(self.p, 1)), W.shape)
            V[:, self.active] = rng.normal(0, .1/np.sqrt(self.hidden),
                                         (self.hidden, self.nactive))
        for j in range(self.t):
            if self.active[j]:
                m = self.target_mean[j]
                c[j] = np.log(m/(1-m)) if self.kinds[j] == "binary" else m
        return np.concatenate([x.ravel() for x in parts])

    def loss_gradient(self, theta):
        W, b, V, c, *a = self.unpack(theta)
        H = np.tanh(self.X @ W + b)
        eta = H @ V + c
        for j in range(self.t):
            eta[:, j] += self.Z[j] @ a[j]
        derivative = np.zeros_like(eta)
        loss = 0.
        for j, kind in enumerate(self.kinds):
            ix = self.indices[j]
            if not len(ix):
                continue
            y, z = self.Y[ix, j], eta[ix, j]
            u = self.w[ix]/(self.norm[j]*self.nactive)
            if kind == "continuous":
                loss += np.dot(u, (z-y)**2)
                derivative[ix, j] = 2*u*(z-y)
            else:
                u = u/self.null_entropy[j]
                loss += np.dot(u, np.logaddexp(0., z)-y*z)
                derivative[ix, j] = u*(expit(z)-y)
        hidden_derivative = (derivative @ V.T)*(1-H**2)
        head_penalty = self.l2/self.nactive
        gradients = [self.X.T @ hidden_derivative + self.l2*W,
                     hidden_derivative.sum(axis=0),
                     H.T @ derivative + head_penalty*V,
                     derivative.sum(axis=0)]
        gradients += [self.Z[j].T @ derivative[:, j] + head_penalty*a[j]
                      for j in range(self.t)]
        loss += .5*self.l2*np.sum(W**2)
        loss += .5*head_penalty*(np.sum(V**2)+sum(np.sum(v**2) for v in a))
        return float(loss), np.concatenate([x.ravel() for x in gradients])


def objective_gradient(theta, X, Z, Y, mask, kinds, weights=None, hidden=4, l2=.01):
    """Convenience function for finite-difference/cross-head checks."""
    return Objective(X, Z, Y, mask, kinds, weights, hidden, l2).loss_gradient(theta)


def fit_joint(X, Z, Y, mask, kinds, weights=None, seed=17, hidden=4,
              l2=.01, task_names=None, maxiter=1000, gtol=1e-6, ftol=1e-10):
    """Fit once. Caller must inspect model['converged']; failure is never hidden.

    The same API supports an independent network by passing a single task.
    hidden=0 supplies a linear skip-only comparator, using whatever Z the caller
    predefines. Reusing a shared-task model to initialise independent fits is not
    supported; each fit uses its declared seed and only its supplied training set.
    """
    obj = Objective(X, Z, Y, mask, kinds, weights, hidden, l2)
    if task_names is None:
        task_names = ["task_" + str(j) for j in range(obj.t)]
    if len(task_names) != obj.t or len(set(task_names)) != obj.t:
        raise ValueError("task_names must uniquely name every task")
    initial = obj.initial(seed)
    initial_loss = obj.loss_gradient(initial)[0]
    result = minimize(obj.loss_gradient, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter":int(maxiter), "gtol":gtol,
                               "ftol":ftol, "maxls":40})
    loss, grad = obj.loss_gradient(result.x)
    parts = [v.copy() for v in obj.unpack(result.x)]
    finite = bool(np.isfinite(result.x).all() and np.isfinite(loss) and np.isfinite(grad).all())
    return {"version":VERSION, "W":parts[0], "b":parts[1], "V":parts[2],
            "c":parts[3], "a":parts[4:], "kinds":obj.kinds,
            "task_names":list(task_names), "active":obj.active.copy(),
            "n_features":obj.p, "skip_features":obj.q,
            "hidden":obj.hidden, "l2":obj.l2, "seed":int(seed),
            "training_null_entropy":list(obj.null_entropy),
            "converged":bool(result.success and finite),
            "optimizer_status":int(result.status), "optimizer_message":str(result.message),
            "iterations":int(result.nit), "function_evaluations":int(result.nfev),
            "initial_objective":float(initial_loss), "objective":loss,
            "max_abs_gradient":float(np.max(np.abs(grad))) if len(grad) else 0.}


def predict_joint(model, X, Z, logits=False):
    """Return n×tasks; unsupported tasks are NaN, binary heads are probabilities.

    Continuous predictions remain in training-standardised units. Caller applies
    the training target inverse transform and evaluates against correctly timed
    held-out observations. logits=True is for numerical validation only.
    """
    X = np.asarray(X, dtype=float)
    Z = [np.asarray(z, dtype=float) for z in Z]
    if X.ndim != 2 or X.shape[1] != model["n_features"] or len(Z) != len(model["kinds"]):
        raise ValueError("prediction dimensions disagree with fitted model")
    if not np.isfinite(X).all() or any(z.shape != (len(X), q) or not np.isfinite(z).all()
                                     for z, q in zip(Z, model["skip_features"])):
        raise ValueError("prediction arrays require fitted dimensions and finite values")
    eta = np.tanh(X @ model["W"] + model["b"]) @ model["V"] + model["c"]
    for j, kind in enumerate(model["kinds"]):
        eta[:, j] += Z[j] @ model["a"][j]
        if not model["active"][j]:
            eta[:, j] = np.nan
        elif kind == "binary" and not logits:
            eta[:, j] = expit(eta[:, j])
    return eta
