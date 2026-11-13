"""Compatibility helper used by the original critic-only EasyBCQ adapter.

The learner kernels are retained verbatim. The original multi-purpose runner
is deliberately not imported: it contains machine-specific experiment logic.
"""
import numpy as np
import torch


def check_proposal_scalers(learner, proposal):
    """Networks retain their own saved scalers, fitted on the same train data."""
    if proposal is None:
        return
    normalizer = proposal.normalize if isinstance(proposal.normalize, torch.nn.Module) else proposal
    for name in ("mean", "scale"):
        expected = np.asarray(getattr(learner, name))
        actual = getattr(normalizer, name).detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
