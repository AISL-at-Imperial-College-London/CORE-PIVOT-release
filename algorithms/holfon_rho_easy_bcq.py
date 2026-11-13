"""An alternative HOFLON formulation inspired by one-step EasyBCQ.

Reuse the frozen truncated-normal behavior policy and single behavior-value
critic. Sample M candidate actions, choose the highest-Q candidate as the
anchor, then maximize Q within an L2 ball of radius rho around that anchor,
intersected with the normalized action bounds [-1, 1]^3.

This adds constrained action refinement to EasyBCQ's sampled-action selection;
it is not part of the algorithm proposed by Brandfonbrener et al. (2021).
No additional networks are trained. In this environment, normalized actions
map to physical actions by multiplication by 5, so the physical L2 refinement
radius is 5 * rho. Failed or unsafe refinement falls back to the anchor;
nondecreasing predicted Q does not guarantee improved environment return.
"""
from dataclasses import dataclass
import copy
import numpy as np
import torch
from algorithms.one_step_easy_bcq import OneStepEasyBCQ


@dataclass(frozen=True)
class OneStepHOFLONConfig:
    radius: float = 0.01
    candidates: int = 50
    search_iterations: int = 40


def refine_action(critic, observation, anchor, radius, iterations):
    """Maximize single-Q in box intersected with a ball; retain a safe anchor."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Radius must be finite and nonnegative.')
    anchor = np.asarray(anchor, dtype=np.float32)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or np.any(np.abs(anchor) > 1):
        raise ValueError('Expected a feasible normalized anchor.')
    if radius == 0:
        return anchor.copy()
    from scipy.optimize import minimize

    def objective(action):
        with torch.enable_grad():
            tensor = torch.tensor(action, dtype=torch.float32,
                                  device=observation.device, requires_grad=True)
            value = critic(observation[None], tensor[None]).sum()
            gradient, = torch.autograd.grad(value, tensor)
        return -float(value.detach()), -gradient.detach().cpu().numpy().astype(np.float64)

    center = anchor.astype(np.float64)
    constraint = dict(type='ineq',
        fun=lambda action: radius**2 - np.square(action-center).sum(),
        jac=lambda action: -2*(action-center))
    solution = minimize(objective, center, jac=True, method='SLSQP',
        bounds=[(-1.,1.)]*3, constraints=[constraint],
        options=dict(maxiter=iterations, ftol=1e-8))
    if not solution.success or not np.isfinite(solution.x).all():
        return anchor.copy()
    result = np.asarray(solution.x, dtype=np.float32)
    # Check the actual float32 action that will be executed, not just the solver's float64 result.
    if (np.any(np.abs(result) > 1)
            or np.linalg.norm(result.astype(np.float64)-center) > radius
            or not np.isfinite(objective(result)[0])
            or objective(result)[0] > objective(anchor)[0]):
        return anchor.copy()
    return result


class OneStepHOFLON:
    """Isolated deployment wrapper; intentionally has no update method."""
    def __init__(self, learner, config=OneStepHOFLONConfig()):
        if not isinstance(learner, OneStepEasyBCQ):
            raise TypeError('Requires a loaded paper-configured OneStepEasyBCQ.')
        if (not np.isfinite(config.radius) or config.radius < 0
                or config.candidates < 1 or config.search_iterations < 1):
            raise ValueError('Invalid radius, candidate count or search budget.')
        self.config = config
        self.device = learner.device
        # Isolation ensures evaluation cannot change the original learner's RNG or modules.
        self.proposal = copy.deepcopy(learner.proposal)
        self.critic = copy.deepcopy(learner.critic)
        self.modules = dict(proposal=self.proposal, critic=self.critic)
        for module in self.modules.values():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.generator = torch.Generator()
        self.generator.set_state(learner.generator.get_state())

    def predict(self, observations):
        observation = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=self.device)
        if observation.shape != (25,) or not torch.isfinite(observation).all():
            raise ValueError('Expected one finite 25-feature observation.')
        with torch.no_grad():
            repeated = observation[None].expand(self.config.candidates, -1)
            actions = self.proposal.distribution(repeated).sample(self.generator)
            anchor = actions[self.critic(repeated, actions).argmax()].cpu().numpy()
        return refine_action(self.critic, observation, anchor,
                             self.config.radius, self.config.search_iterations)
