from __future__ import annotations

from .instance_reduction import (
    DROP3,
    FCNN3,
    RNG2,
    RNGE2,
    InstanceReducer,
    get_instance_reducer,
)
from .knn import KNNClassifier, kNNAlgorithm
from .random_forest import rfAlgorithm
from .retention import (
    AlwaysRetain,
    CaseBase,
    DifferentClass,
    NeverRetain,
    OblivionByGoodness,
    RetentionPolicy,
    get_retention_policy,
)
from .svm import svmAlgorithm
from .voting import (
    InverseDistanceWeightedVoting,
    MajorityVoting,
    VotingStrategy,
    get_voting_strategy,
)

__all__ = [
    "DROP3",
    "FCNN3",
    "RNG2",
    "RNGE2",
    "AlwaysRetain",
    "CaseBase",
    "DifferentClass",
    "InstanceReducer",
    "InverseDistanceWeightedVoting",
    "KNNClassifier",
    "MajorityVoting",
    "NeverRetain",
    "OblivionByGoodness",
    "RetentionPolicy",
    "VotingStrategy",
    "get_instance_reducer",
    "get_retention_policy",
    "get_voting_strategy",
    "kNNAlgorithm",
    "rfAlgorithm",
    "svmAlgorithm",
]
