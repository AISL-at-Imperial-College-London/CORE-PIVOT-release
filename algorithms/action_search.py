"""Optional, guarded local action refinement; always retain a feasible anchor."""
import numpy as np
import torch


def refine_action(critic, observation, anchor, *, radius=None, max_iterations=40):
    if radius is not None and radius < 0:
        raise ValueError('Radius must be nonnegative.')
    if radius == 0:
        return np.asarray(anchor, dtype=np.float32).copy()
    try:
        from scipy.optimize import minimize
    except ImportError as error:
        raise ImportError('Local action refinement requires pip install -e ".[training]".') from error
    device = observation.device
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or np.abs(anchor).max() > 1:
        raise ValueError('Expected a finite feasible normalized anchor.')

    def objective(action):
        with torch.enable_grad():
            action_tensor = torch.tensor(action, dtype=torch.float32, device=device,
                                         requires_grad=True)
            value = critic.minimum(observation[None, :], action_tensor[None, :]).sum()
            gradient, = torch.autograd.grad(value, action_tensor)
        return -float(value.detach()), -gradient.detach().cpu().numpy().astype(np.float64)

    constraints = () if radius is None else ({
        'type': 'ineq', 'fun': lambda action: radius ** 2 - np.square(action - anchor).sum(),
        'jac': lambda action: -2 * (action - anchor)},)
    solution = minimize(objective, anchor, jac=True, bounds=[(-1., 1.)] * 3,
                        constraints=constraints, method='SLSQP',
                        options={'maxiter': max_iterations, 'ftol': 1e-8})
    result = np.asarray(solution.x)
    feasible = (solution.success and np.isfinite(result).all()
                and np.abs(result).max() <= 1 + 1e-7
                and (radius is None or np.linalg.norm(result - anchor) <= radius + 1e-7))
    if feasible:
        result = np.clip(result, -1, 1).astype(np.float32)
        # Rounding must not introduce a constraint violation or lower Q.
        feasible = ((radius is None or np.linalg.norm(result - anchor) <= radius + 1e-7)
                    and objective(result)[0] <= objective(anchor)[0])
    return result if feasible else anchor.astype(np.float32)
