"""TRIAGE: template-level risk-informed active review for an LLM log parser with a template cache.

    from triage import Triage, TransferPrior

    prior = TransferPrior(source_states, source_labels, kind="mlp", hp={...}, w=100)   # trained on other systems
    t = Triage(prior, M=20)
    t.set_groups(H_first, sizes, contents)        # first-occurrence hidden states, lines per group so far
    for gid in t.queue(5):                        # next templates to show the reviewer
        ...
    t.update({gid: verdict, ...})                 # verdict: True if the cached template was correct
    t.set_groups(...)                             # call again when new groups appear or sizes grow

Components (each can be switched off for ablations): zero-label prior (cross-system transfer model, refit with
verdicts), Clotho-style balanced selection (uncertainty + diversity among the top-M by priority), propagation
(a verdict covers the whole Drain group; the caller passes `sizes` accordingly), frequency weighting
(priority = risk x lines so far).
"""
from .prior import TransferPrior, OwnProbe
from .core import Triage

__all__ = ["Triage", "TransferPrior", "OwnProbe"]
