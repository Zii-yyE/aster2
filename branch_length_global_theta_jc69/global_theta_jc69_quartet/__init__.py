"""Minimal global-theta JC69 quartet-stitching estimator for aster2 integration."""

__all__ = [
    "BranchAggregate",
    "EstimatedNtaxaResult",
    "WitnessSpec",
    "build_witness_map",
    "estimate_branch_lengths",
]


def __getattr__(name):
    if name in __all__:
        from . import global_theta_jc69_estimator as estimator
        return getattr(estimator, name)
    raise AttributeError(name)
