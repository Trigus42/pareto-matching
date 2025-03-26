"""Pareto-efficient matching for the stable roommates problem."""

from pareto_matching.matching import (
    find_pareto_efficient_matching, 
    verify_pareto_efficiency,
    MatchingResult,
    Person,
    PreferenceList
)

__all__ = [
    "find_pareto_efficient_matching", 
    "verify_pareto_efficiency",
    "MatchingResult",
    "Person",
    "PreferenceList"
]