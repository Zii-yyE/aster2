"""
JC69 symbolic substitution model used by the global-theta quartet estimator.

This integration package intentionally exposes only JC69. Branch lengths are in
substitution units, and the global MSC/substitution scale parameter is
``theta = 4 Ne mu``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import sympy as sp


StateDist = Tuple[sp.Expr, ...]
STATE_ORDER = ("A", "C", "G", "T")
STATE_TO_INDEX = {state: index for index, state in enumerate(STATE_ORDER)}


class SymbolicSubstitutionModel(ABC):
    name: str = "symbolic-model"

    @property
    @abstractmethod
    def state_labels(self) -> Tuple[str, ...]:
        pass

    @property
    @abstractmethod
    def stationary_distribution(self) -> StateDist:
        pass

    @abstractmethod
    def transition_probability(self, i: int, j: int, branch_length: sp.Expr) -> sp.Expr:
        pass

    @abstractmethod
    def normalize_observation(self, symbol: str) -> Optional[str]:
        pass

    @property
    def state_count(self) -> int:
        return len(self.state_labels)

    @property
    def state_to_index(self) -> Mapping[str, int]:
        return {state: index for index, state in enumerate(self.state_labels)}


@dataclass(frozen=True)
class JC69SubstitutionModel(SymbolicSubstitutionModel):
    """JC69 with off-diagonal rate ``mu/3`` and diagonal rate ``-mu``."""

    mu: sp.Expr = sp.Integer(1)
    name: str = "jc69"

    def __post_init__(self) -> None:
        mu = sp.sympify(self.mu)
        if mu.is_positive is False:
            raise ValueError("JC69 requires mu > 0.")
        object.__setattr__(self, "mu", mu)

    @property
    def state_labels(self) -> Tuple[str, ...]:
        return STATE_ORDER

    @property
    def stationary_distribution(self) -> StateDist:
        q = sp.Rational(1, 4)
        return (q, q, q, q)

    def normalize_observation(self, symbol: str) -> Optional[str]:
        key = symbol.strip().upper()
        if key in STATE_TO_INDEX:
            return key
        return None

    def transition_probability(self, i: int, j: int, branch_length: sp.Expr) -> sp.Expr:
        t = sp.sympify(branch_length)
        e = sp.exp(-(sp.Rational(4, 3) * self.mu) * t)
        if i == j:
            return sp.Rational(1, 4) + sp.Rational(3, 4) * e
        return sp.Rational(1, 4) - sp.Rational(1, 4) * e


def build_symbolic_model(
    model_name: str = "JC69",
    model_params: Optional[Mapping[str, sp.Expr | float | int]] = None,
) -> SymbolicSubstitutionModel:
    params = dict(model_params or {})
    key = model_name.strip().lower()
    if key != "jc69":
        raise ValueError("This integration package supports only JC69.")
    return JC69SubstitutionModel(mu=sp.sympify(params.get("mu", 1)))
