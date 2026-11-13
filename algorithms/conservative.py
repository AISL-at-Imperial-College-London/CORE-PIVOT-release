"""Importance-sampled CQL regularizer, with densities in normalized action space."""
import math

import torch


def conservative_penalty(critic, observations, next_observations, data_actions,
                         proposal, generator, samples=10, temperature=1.):
    batch_size, dimensions = observations.shape
    repeated = observations[:, None, :].expand(-1, samples, -1).reshape(-1, dimensions)
    next_repeated = next_observations[:, None, :].expand(-1, samples, -1).reshape(-1, dimensions)
    with torch.no_grad():
        current_actions, current_logp = proposal.sample(repeated, generator)
        next_actions, next_logp = proposal.sample(next_repeated, generator)
        uniform_actions = 2 * torch.rand((len(repeated), 3), generator=generator,
                                         device=observations.device) - 1
    # All Qs are evaluated at the CURRENT state, even for next-state proposals.
    candidate_actions = (uniform_actions, current_actions, next_actions)
    densities = (torch.full_like(current_logp, -3 * math.log(2)), current_logp, next_logp)
    corrected = [[], []]
    for actions, logp in zip(candidate_actions, densities):
        values = critic(repeated, actions)
        for head, value in zip(corrected, values):
            head.append((value - logp).reshape(batch_size, samples))
    recorded_q = critic(observations, data_actions)
    losses = []
    for head, recorded in zip(corrected, recorded_q):
        values = torch.cat(head, dim=1)
        log_integral = temperature * torch.logsumexp(values / temperature, dim=1)
        # Averaging the mixture changes only a constant for a fixed CQL weight.
        log_integral = log_integral - temperature * math.log(3 * samples)
        losses.append(log_integral.mean() - recorded.mean())
    return sum(losses)
