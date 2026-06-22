from __future__ import annotations

import numpy as np
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _empty_dataset_dict():
    return {
        "ACSIncome": {},
        "ACSPublicCoverage": {},
        "ACSTravelTime": {},
        "Wilds": {},
        "Amazon": {},
        "Scaled_Pubcov": {},
    }


@dataclass
class ExperimentContext:
    dataset: str

    sorted_train_sources: Dict[str, Any] = field(default_factory=dict)
    test_source: Dict[str, Any] = field(default_factory=lambda: {"Wilds": []})
    comb: Dict[str, Any] = field(default_factory=dict)
    scaler: Dict[str, Any] = field(default_factory=dict)

    source_X: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    source_y: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    source_sens: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)

    test_X_scaled: Dict[str, np.ndarray] = field(default_factory=dict)
    test_y: Dict[str, np.ndarray] = field(default_factory=dict)
    test_sens_masks: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)

    surr_model: Dict[str, Any] = field(default_factory=dict)
    surr_scaler: Dict[str, Any] = field(default_factory=dict)
    g_val: Dict[str, Any] = field(default_factory=dict)
    norm_g_sources: Dict[str, Any] = field(default_factory=dict)

    amazon_source_cats: List[str] = field(default_factory=list)
    amazon_target_cats: List[str] = field(default_factory=list)
    amazon_source_sources: Dict[str, Any] = field(default_factory=dict)
    amazon_target_sources: Dict[str, Any] = field(default_factory=dict)

    source_pool: Dict[str, Any] = field(default_factory=dict)
    test_dict: Dict[str, Any] = field(default_factory=dict)

    kurt: Any = None
    prev: Any = None

    num_classes: int = 0

    amazon_target_stats: Optional[Dict[str, Any]] = None

    source_list: List[int] = field(default_factory=list)
    gain_lambda: float = 10.0
    cost_type: str = "zero"
    limit: int = 100_000
    k_max: int = 15
    method: str = "normal"
    random_gene: bool = False

    covered: Dict[Any, int] = field(default_factory=dict)
    metrics_dict: Dict[str, dict] = field(default_factory=_empty_dataset_dict)
    dp_dict: Dict[str, dict] = field(default_factory=_empty_dataset_dict)
    pca_cache: Dict[str, Any] = field(default_factory=dict)

    profits: List[List[Any]] = field(default_factory=lambda: [[0, 0, 0]])
    subset_array: List[Any] = field(default_factory=list)
    algo_over: bool = False
    train_time: float = 0.0
    bottleneck: bool = False
    only_accuracy: bool = False

    _cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_cache_lock":
                object.__setattr__(result, k, threading.Lock())
            else:
                object.__setattr__(result, k, copy.deepcopy(v, memo))
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_cache_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()

    def get_metric(self, key: Any):
        with self._cache_lock:
            return self.metrics_dict[self.dataset].get(key)

    def set_metric(self, key: Any, value: Any):
        with self._cache_lock:
            self.metrics_dict[self.dataset][key] = value
            self.covered[key] = 1

    def has_metric(self, key: Any):
        with self._cache_lock:
            return key in self.metrics_dict[self.dataset]

    def get_dp(self, key: Any):
        with self._cache_lock:
            return self.dp_dict[self.dataset].get(key)

    def set_dp(self, key: Any, value: Any):
        with self._cache_lock:
            self.dp_dict[self.dataset][key] = value

    def has_dp(self, key: Any):
        with self._cache_lock:
            return key in self.dp_dict[self.dataset]

    def mark_covered(self, key: Any):
        with self._cache_lock:
            self.covered[key] = 1

    def covered_count(self):
        with self._cache_lock:
            return len(self.covered)

    def reset_for_run(self):
        with self._cache_lock:
            self.covered = {}
            self.metrics_dict[self.dataset] = {}
            self.dp_dict[self.dataset] = {}
            self.profits = [[0, 0, 0]]
            self.algo_over = False
            self.train_time = 0.0


    @property
    def is_regression(self):
        return self.dataset == "ACSTravelTime"

    @property
    def is_fairness_dataset(self):
        return self.dataset not in ("ACSTravelTime", "Scaled_Pubcov")

    @property
    def target_col(self):
        mapping = {
            "ACSIncome": "PINCP",
            "ACSPublicCoverage": "PUBCOV",
            "Scaled_Pubcov": "PUBCOV",
            "ACSTravelTime": "JWMNP",
        }
        return mapping.get(self.dataset, "")

    @property
    def sensitive_col(self):
        if self.dataset == "ACSIncome":
            return "SEX"
        if self.dataset in ("ACSPublicCoverage", "Scaled_Pubcov"):
            return "RAC1P"
        return None

    @property
    def drop_cols(self):
        base = ["Year", "State", self.target_col]
        if self.dataset not in ("Wilds", "Amazon"):
            base.append("Source ID")
        return base
