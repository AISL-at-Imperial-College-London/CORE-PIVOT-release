"""The corrected SLSQP fallback used by the paper's CPU evaluations.

HOFLON keeps a feasible improving evaluated iterate on status 9 only. CORE
retains its original action_search path for every other exit. These are not
global optimisers. Run separate processes, not threads: the legacy CORE hook
temporarily intercepts scipy.optimize.minimize to record evaluated iterates.
"""
import numpy as np
import torch
from scipy.optimize import minimize


def refine_action(critic, observation, anchor, radius, iterations):
    """Single-Q radius refinement; radius=None is the box-only ablation."""
    if radius is not None and (not np.isfinite(radius) or radius < 0):
        raise ValueError('Invalid radius')
    anchor = np.asarray(anchor, dtype=np.float32)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or np.any(np.abs(anchor) > 1):
        raise ValueError('Invalid anchor')
    if radius == 0:
        return anchor.copy()
    center = anchor.astype(np.float64)
    visited = []

    def objective(a):
        visited.append(np.asarray(a, dtype=np.float64).copy())
        with torch.enable_grad():
            x = torch.tensor(a, dtype=torch.float32, device=observation.device, requires_grad=True)
            v = critic(observation[None], x[None]).sum()
            g, = torch.autograd.grad(v, x)
        return -float(v.detach()), -g.detach().cpu().numpy().astype(np.float64)

    def value(a):
        with torch.no_grad():
            return float(critic(observation[None], torch.tensor(a, device=observation.device)[None]).sum())

    def feasible(a):
        a = np.asarray(a, dtype=np.float32)
        return (a.shape == (3,) and np.isfinite(a).all() and np.all(np.abs(a) <= 1)
                and (radius is None or np.linalg.norm(a.astype(np.float64) - center) <= radius))

    solution = minimize(objective, center, jac=True, method='SLSQP', bounds=[(-1., 1.)] * 3,
        constraints=[] if radius is None else [dict(type='ineq',
            fun=lambda a: radius ** 2 - np.square(a - center).sum(), jac=lambda a: -2 * (a - center))],
        options=dict(maxiter=iterations, ftol=1e-8))
    baseline = value(anchor)
    if not np.isfinite(baseline):
        raise ValueError('Nonfinite anchor Q')
    if solution.success:
        a = np.asarray(solution.x, dtype=np.float32)
        return a if feasible(a) and np.isfinite(value(a)) and value(a) >= baseline else anchor.copy()
    if solution.status != 9:
        return anchor.copy()
    best, bestq = anchor.copy(), baseline
    for a in visited + [solution.x]:
        a = np.asarray(a, dtype=np.float32)
        if feasible(a):
            q = value(a)
            if np.isfinite(q) and q > bestq:
                best, bestq = a.copy(), q
    return best


def install_core_fallback():
    """Install the exact best-feasible status-9 correction for CPU CORE."""
    import algorithms.core_pivot as cp
    from algorithms.action_search import refine_action as original
    import scipy.optimize as opt
    native = opt.minimize

    def refined(critic, observation, anchor, **kwargs):
        calls = []

        def trace(fun, x0, *a, **kw):
            visited = []

            def wrapped(z):
                ans = fun(z)
                visited.append(np.array(z, copy=True))
                return ans

            sol = native(wrapped, x0, *a, **kw)
            calls.append((sol, visited))
            return sol

        opt.minimize = trace
        try:
            baseline = original(critic, observation, anchor, **kwargs)
        finally:
            opt.minimize = native
        if not calls or calls[0][0].status != 9:
            return baseline
        assert kwargs.get('radius') is None
        sol, visited = calls[0]

        def value(a):
            with torch.no_grad():
                return float(critic.minimum(observation[None], torch.tensor(a, dtype=torch.float32)[None]).item())

        best = np.array(anchor, dtype=np.float32)
        bestq = value(best)
        assert np.isfinite(bestq)
        for a in visited + [sol.x]:
            a = np.asarray(a)
            if a.shape != (3,) or not np.isfinite(a).all() or (np.abs(a) > 1 + 1e-7).any():
                continue
            a = np.clip(a, -1, 1).astype(np.float32)
            v = value(a)
            if np.isfinite(v) and v > bestq:
                best, bestq = a.copy(), v
        return best

    cp.refine_action = refined
