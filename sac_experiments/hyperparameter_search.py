"""Compatibility facade for the modular ``sac_experiments.search`` package.

New code should import from ``sac_experiments.search``. Existing scripts keep
working through these re-exports.
"""

from sac_experiments.search.artifacts import render_config as _render_config
from sac_experiments.search.config import (
    RevalidationConfig,
    SearchConfig,
    SearchParameter,
    load_search_config,
)
from sac_experiments.search.revalidation import (
    confidence_bounds as _confidence_bounds,
    paired_lower_bound as _paired_lower_bound,
    run_revalidation,
)
from sac_experiments.search.runner import TrialEvaluationReporter, run_search

__all__ = [
    "RevalidationConfig",
    "SearchConfig",
    "SearchParameter",
    "TrialEvaluationReporter",
    "_confidence_bounds",
    "_paired_lower_bound",
    "_render_config",
    "load_search_config",
    "run_revalidation",
    "run_search",
]
