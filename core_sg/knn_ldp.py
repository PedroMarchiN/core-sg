# TODO(phase-1-relocation): everything in this file is a faithful, plain-
# Euclidean kNN-LDP reimplementation that does not use any Core-SG-specific
# machinery (no mutual reachability, no core distances, no MST). It lives
# here temporarily as scaffolding toward Phase 2 (the Core-SG-integrated
# density-aware variant); once Phase 2 exists, this module should move to
# its own separate repository, since it isn't really "Core-SG" on its own.
from __future__ import annotations

import heapq
from itertools import count
from time import time
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data
except ImportError:  # pragma: no cover - exercised only on older sklearn
    validate_data = None

from .knn import knn_from_precomputed

ProgressCallback = Callable[[str, float, dict[str, Any]], None]


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _emit_progress(
    event: str,
    elapsed: float,
    *,
    verbose: int = 0,
    progress_callback: ProgressCallback | None = None,
    message: str | None = None,
    **info: Any,
) -> None:
    # Duplicated from core_sg.py instead of imported: importing core_sg.core_sg
    # pulls in the hdbscan_adapter/MST machinery this module deliberately
    # avoids building for a pure classifier.
    if progress_callback is not None:
        progress_callback(event, elapsed, dict(info))
    if verbose:
        print(message or f"{event} done in {elapsed:.2f}s")


# TODO(phase-1-relocation): move to the standalone repo (see module TODO).
def build_knn_ldp_graph_from_data(
    X: np.ndarray,
    k_max: int,
    *,
    metric: str = "euclidean",
    p: int = 2,
    pairwise_dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the reusable directed k_max-nearest-neighbor structure for kNN-LDP.

    Computes the dense pairwise distance matrix once and extracts, for every
    point, its `k_max` nearest neighbors (self excluded) via
    `knn_from_precomputed`. Unlike `build_core_sg_from_data`, this function
    does not compute core distances and does not build MST/mutual-
    reachability support: kNN-LDP's propagation only needs neighbor
    identities, not the density-based structures clustering needs.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix.
    k_max : int
        Maximum neighborhood size to cache; any `k <= k_max` can later be
        served from the returned arrays without recomputation.
    metric : str, default="euclidean"
        Distance metric used to build the pairwise distances.
    p : int, default=2
        Power parameter for the Minkowski metric.

    Returns
    -------
    idxs_graph : np.ndarray of shape (n_samples, k_max), dtype int64
        `idxs_graph[i, :k]` are the indices of the k nearest neighbors of
        `i`, sorted ascending by distance, for any `k <= k_max`.
    dists_graph : np.ndarray of shape (n_samples, k_max), dtype float64
        Matching neighbor distances.
    """
    X = np.asarray(X)
    n = X.shape[0]

    if n <= 1:
        raise ValueError("X needs to have shape > 1")
    if k_max <= 0 or k_max >= n:
        raise ValueError("k_max invalid (1 <= k_max <= n-1).")

    if metric == "minkowski":
        D = pairwise_distances(X, metric=metric, p=p)
    elif metric == "arccos":
        D = pairwise_distances(X, metric="cosine")
    else:
        D = pairwise_distances(X, metric=metric)

    D = np.ascontiguousarray(D, dtype=pairwise_dtype)
    np.fill_diagonal(D, 0.0)

    idxs_graph, dists_graph = knn_from_precomputed(D, k=k_max, include_self=False)
    return idxs_graph, dists_graph


# TODO(phase-1-relocation): move to the standalone repo (see module TODO).
def build_reverse_knn_index(
    idxs_graph: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the reverse-kNN (RkNN) index for the first `k` columns of `idxs_graph`.

    Parameters
    ----------
    idxs_graph : np.ndarray of shape (n_samples, k_max)
        Cached neighbor indices from `build_knn_ldp_graph_from_data`; only
        `idxs_graph[:, :k]` is read.
    k : int
        Neighborhood size to index, `1 <= k <= idxs_graph.shape[1]`.

    Returns
    -------
    indptr : np.ndarray of shape (n_samples + 1,), dtype int64
    indices : np.ndarray of shape (n_samples * k,), dtype int64
        CSR-style pair such that `indices[indptr[x]:indptr[x + 1]]` are the
        points `p` for which `x` is one of `p`'s `k` nearest neighbors
        (i.e. RkNN(x)).
    """
    idxs_graph = np.asarray(idxs_graph)
    if idxs_graph.ndim != 2:
        raise ValueError("idxs_graph must be 2D (n_samples, k_max).")

    n, k_max = idxs_graph.shape
    if k <= 0 or k > k_max:
        raise ValueError("k invalid (1 <= k <= k_max).")

    neighbor = idxs_graph[:, :k].reshape(-1).astype(np.int64, copy=False)
    source = np.repeat(np.arange(n, dtype=np.int64), k)

    order = np.argsort(neighbor, kind="stable")
    neighbor_sorted = neighbor[order]
    indices = source[order]
    indptr = np.searchsorted(neighbor_sorted, np.arange(n + 1, dtype=np.int64))

    return indptr.astype(np.int64, copy=False), indices.astype(np.int64, copy=False)


# TODO(phase-1-relocation): move to the standalone repo (see module TODO).
class _MaxPriorityQueue:
    """
    Max-priority queue over integer node weights.

    `heapq` has no native decrease/increase-key operation, so updates are
    emulated with the standard lazy-deletion pattern: pushing a fresh entry
    for a node replaces its record in `_entries`, and any older heap entry
    for that node is recognized as stale (and skipped) because it no longer
    matches `_entries[node]`.
    """

    __slots__ = ("_heap", "_entries", "_counter")

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, int]] = []
        self._entries: dict[int, tuple[float, int, int]] = {}
        self._counter = count()

    def push_or_update(self, node: int, weight: float) -> None:
        entry = (-weight, next(self._counter), node)
        self._entries[node] = entry
        heapq.heappush(self._heap, entry)

    def pop_max(self) -> tuple[int, float] | None:
        while self._heap:
            entry = heapq.heappop(self._heap)
            neg_weight, _, node = entry
            if self._entries.get(node) is entry:
                del self._entries[node]
                return node, -neg_weight
        return None

    def remaining_nodes(self) -> list[int]:
        return list(self._entries.keys())

    def __contains__(self, node: int) -> bool:
        return node in self._entries

    def __len__(self) -> int:
        return len(self._entries)


# TODO(phase-1-relocation): move to the standalone repo (see module TODO).
def propagate_knn_ldp(
    idxs_graph: np.ndarray,
    y_encoded: np.ndarray,
    n_classes: int,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run kNN-LDP's label distribution propagation (Algorithm 1) for a given `k`.

    Every instance's distribution is defined over `n_classes` real classes
    plus a synthetic "unknown" outcome (the paper's convention), so every
    row of `label_dist` always sums to 1: an instance starts fully
    "unknown" and is overwritten with its actual distribution once resolved
    (either because it was originally labeled, or because propagation
    reached it).

    Parameters
    ----------
    idxs_graph : np.ndarray of shape (n_samples, k_max)
        Cached neighbor indices from `build_knn_ldp_graph_from_data`; only
        `idxs_graph[:, :k]` is read.
    y_encoded : np.ndarray of shape (n_samples,), dtype int
        Encoded class indices in `[0, n_classes)`, with `-1` for unlabeled.
    n_classes : int
        Number of real classes (excludes the synthetic "unknown" outcome).
    k : int, keyword-only
        Neighborhood size to use, `1 <= k <= idxs_graph.shape[1]`.

    Returns
    -------
    label_dist : np.ndarray of shape (n_samples, n_classes + 1)
        Final label probability distribution per point; column `n_classes`
        is the "unknown"/abstention mass.
    abstained : np.ndarray of shape (n_samples,), dtype bool
        True for points whose entire mass ended up on "unknown", i.e. no
        labeled instance was reachable through the directed kNN graph.
    """
    idxs_graph = np.asarray(idxs_graph)
    if idxs_graph.ndim != 2:
        raise ValueError("idxs_graph must be 2D (n_samples, k_max).")

    n, k_max = idxs_graph.shape
    if k <= 0 or k > k_max:
        raise ValueError("k invalid (1 <= k <= k_max).")
    if n_classes <= 0:
        raise ValueError("n_classes must be >= 1.")

    y_encoded = np.asarray(y_encoded, dtype=np.int64)
    if y_encoded.shape != (n,):
        raise ValueError(f"y_encoded must have shape ({n},), got {y_encoded.shape}.")

    resolved = y_encoded != -1
    if not np.any(resolved):
        raise ValueError(
            "y_encoded must contain at least one labeled instance (value != -1)."
        )

    unknown_col = n_classes
    n_outcomes = n_classes + 1

    label_dist = np.zeros((n, n_outcomes), dtype=np.float64)
    label_dist[:, unknown_col] = 1.0
    label_dist[resolved, unknown_col] = 0.0
    label_dist[resolved, y_encoded[resolved]] = 1.0

    neighbors = idxs_graph[:, :k]
    running_sum = label_dist[neighbors].sum(axis=1)

    indptr, rindices = build_reverse_knn_index(idxs_graph, k)

    pq = _MaxPriorityQueue()
    for node in np.flatnonzero(~resolved):
        weight = running_sum[node, :n_classes].sum()
        pq.push_or_update(int(node), float(weight))

    abstained = np.zeros(n, dtype=bool)

    while len(pq) > 0:
        popped = pq.pop_max()
        if popped is None:
            break
        node, weight = popped

        if weight > 0.0:
            old_dist = label_dist[node].copy()
            label_dist[node] = running_sum[node] / k
            resolved[node] = True

            delta = label_dist[node] - old_dist
            ps = rindices[indptr[node] : indptr[node + 1]]
            if ps.size:
                running_sum[ps] += delta
                new_weights = running_sum[ps, :n_classes].sum(axis=1)
                for p, new_weight in zip(ps.tolist(), new_weights.tolist()):
                    if p in pq:
                        pq.push_or_update(p, new_weight)
        else:
            # Max-heap invariant: weights only ever increase as neighbors
            # resolve, so once the current maximum weight is 0, every
            # remaining queued node's weight is also 0 — the whole
            # remainder abstains at once instead of being popped one by one.
            abstain_idx = np.array(pq.remaining_nodes() + [node], dtype=np.int64)
            label_dist[abstain_idx, :] = 0.0
            label_dist[abstain_idx, unknown_col] = 1.0
            abstained[abstain_idx] = True
            resolved[abstain_idx] = True
            break

    return label_dist, abstained


# TODO(phase-1-relocation): move to the standalone repo (see module TODO).
class KNNLDP:
    """
    Reusable kNN-LDP propagation engine.

    `fit(X, k_max)` builds the directed k_max-nearest-neighbor structure once
    (via `build_knn_ldp_graph_from_data`). `propagate(y, k=...)` re-runs
    Algorithm 1's priority-queue propagation for any `k <= k_max` and any
    label vector without recomputing pairwise distances or the k_max-NN
    graph.
    """

    def __init__(
        self,
        metric: str = "euclidean",
        p: int = 2,
        verbose: int = 0,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.metric = metric
        self.p = p
        self.verbose = verbose
        self.progress_callback = progress_callback

        self.n_samples_ = None
        self.k_max_ = None
        self.idxs_graph_ = None
        self.dists_graph_ = None

        self.k_ = None
        self.classes_ = None
        self.label_distributions_extended_ = None
        self.abstained_ = None

    def fit(self, X: np.ndarray, k_max: int) -> "KNNLDP":
        """
        Build the reusable k_max-nearest-neighbor structure.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        k_max : int
            Maximum neighborhood size used to build the reusable structure.

        Returns
        -------
        KNNLDP
            The fitted instance itself.
        """
        X = np.asarray(X)

        t0 = time()
        idxs_graph, dists_graph = build_knn_ldp_graph_from_data(
            X, k_max, metric=self.metric, p=self.p
        )
        t1 = time()
        _emit_progress(
            "build",
            t1 - t0,
            verbose=self.verbose,
            progress_callback=self.progress_callback,
            message=f"kNN-LDP graph build done in {t1 - t0:.2f}s",
            k_max=k_max,
        )

        self.idxs_graph_ = idxs_graph
        self.dists_graph_ = dists_graph
        self.n_samples_ = X.shape[0]
        self.k_max_ = k_max
        return self

    def _ensure_fitted(self) -> None:
        if self.k_max_ is None:
            raise AttributeError("KNNLDP is not fitted yet. Run fit first.")

    def _validate_k(self, k: int) -> None:
        self._ensure_fitted()
        if k <= 0 or k > self.k_max_:
            raise ValueError("k invalid (1 <= k <= k_max).")

    def propagate(self, y: np.ndarray, *, k: int) -> "KNNLDP":
        """
        Re-run Algorithm 1 for `k` and `y` using the cached kNN structure.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Semi-supervised target with `-1` for unlabeled samples.
        k : int, keyword-only
            Neighborhood size, `1 <= k <= k_max`.

        Returns
        -------
        KNNLDP
            The instance itself, with `label_distributions_extended_`,
            `abstained_`, `classes_`, and `k_` updated in place.
        """
        self._validate_k(k)

        y = np.asarray(y)
        if y.shape != (self.n_samples_,):
            raise ValueError(f"y must have shape ({self.n_samples_},), got {y.shape}.")

        classes = np.unique(y[y != -1])
        if classes.size == 0:
            raise ValueError(
                "y must contain at least one labeled instance (value != -1)."
            )

        y_encoded = np.full(self.n_samples_, -1, dtype=np.int64)
        mask = y != -1
        y_encoded[mask] = np.searchsorted(classes, y[mask])

        t0 = time()
        label_dist, abstained = propagate_knn_ldp(
            self.idxs_graph_, y_encoded, classes.size, k=k
        )
        t1 = time()
        _emit_progress(
            "propagate",
            t1 - t0,
            verbose=self.verbose,
            progress_callback=self.progress_callback,
            message=f"kNN-LDP propagate K = {k} done in {t1 - t0:.2f}s",
            k=k,
        )

        self.classes_ = classes
        self.k_ = k
        self.label_distributions_extended_ = label_dist
        self.abstained_ = abstained
        return self


# TODO(phase-1-relocation): move to the standalone repo (see module TODO),
# unless/until Phase 2 grows this same class in place with a Core-SG-backed
# neighbor space, at which point re-evaluate whether it stays here instead.
class KNNLDPClassifier(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn-style semi-supervised classifier implementing kNN-LDP
    (Gøttcke, Zimek, Campello, 2025), with the k_max-nearest-neighbor
    structure built once and reused across repeated calls with different
    `k`/`y`.

    The internal `KNNLDP` engine is built only once, on the first `fit(...)`
    call, at the configured `k_max`. Unlike a plain kNN classifier, `y` is
    not discarded after use: every `fit(...)` call re-runs the label
    propagation for the given `y` and `k`, while the k_max-nearest-neighbor
    structure is rebuilt only if it does not exist yet.

    `y` follows the `-1`-for-unlabeled convention used by
    `sklearn.semi_supervised.LabelPropagation`, so real class labels must be
    non-negative. Predictions are transductive only in this version:
    `predict(X)`/`predict_proba(X)` return the cached result for the
    training set when `X` is `None` or matches the fitted training data in
    shape; a genuinely new `X` raises `NotImplementedError` (inductive
    prediction on unseen query points is not implemented yet). This keeps
    `fit(X, y).predict(X)` always equal to `fit(X, y).transduction_`, unlike
    `sklearn.semi_supervised.LabelPropagation`, whose `predict(X)` performs
    true induction and is not guaranteed to match its own `transduction_`.

    Points that cannot be reached from any labeled instance abstain: their
    predicted label is `-1` (the same sentinel used for unlabeled input) and
    their `predict_proba` row does not sum to 1, so standard scikit-learn
    classification metrics score them as errors with no extra handling
    required, matching the evaluation convention of the original paper.
    """

    def __init__(
        self,
        k_max: int,
        metric: str = "euclidean",
        p: int = 2,
        verbose: int = 0,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.k_max = k_max
        self.metric = metric
        self.p = p
        self.verbose = verbose
        self.progress_callback = progress_callback

    def _validate_X(self, X: Any) -> np.ndarray:
        check_params = {
            "accept_sparse": False,
            "ensure_2d": True,
            "ensure_min_samples": 2,
            "dtype": [np.float64, np.float32],
        }
        if validate_data is not None:
            return validate_data(self, X, y="no_validation", reset=True, **check_params)

        X_checked = check_array(X, **check_params)
        self.n_features_in_ = X_checked.shape[1]
        return X_checked

    def _validate_k_max(self, n_samples: int) -> int:
        if not _is_integer(self.k_max):
            raise ValueError("k_max must be an integer.")

        k_max = int(self.k_max)
        if k_max < 1:
            raise ValueError("k_max must be >= 1.")
        if k_max >= n_samples:
            raise ValueError("k_max must satisfy 1 <= k_max <= n_samples - 1.")
        return k_max

    @staticmethod
    def _validate_k(k: int | None, *, k_max: int) -> int:
        if k is None:
            return k_max
        if not _is_integer(k):
            raise ValueError("k must be an integer or None.")

        k_value = int(k)
        if k_value < 1:
            raise ValueError("k must be >= 1.")
        if k_value > k_max:
            raise ValueError("k must satisfy 1 <= k <= k_max.")
        return k_value

    @staticmethod
    def _validate_y(y: Any, *, n_samples: int) -> np.ndarray:
        y_checked = np.asarray(y)
        if y_checked.shape != (n_samples,):
            raise ValueError(
                f"y must have shape ({n_samples},), got {y_checked.shape}."
            )
        if not np.any(y_checked != -1):
            raise ValueError(
                "y must contain at least one labeled instance (value != -1)."
            )
        return y_checked

    def _engine_kwargs(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "p": self.p,
            "verbose": self.verbose,
            "progress_callback": self.progress_callback,
        }

    def _has_engine(self) -> bool:
        return isinstance(getattr(self, "knn_ldp_", None), KNNLDP)

    @staticmethod
    def _labels_from_distributions(
        label_distributions: np.ndarray, classes: np.ndarray, abstained: np.ndarray
    ) -> np.ndarray:
        idx = np.argmax(label_distributions, axis=1)
        labels = classes[idx]
        return np.where(abstained, -1, labels)

    def _sync_current_outputs(self) -> None:
        engine = self.knn_ldp_
        self.classes_ = engine.classes_
        self.label_distributions_extended_ = engine.label_distributions_extended_
        self.label_distributions_ = engine.label_distributions_extended_[:, :-1]
        self.abstained_ = engine.abstained_
        self.transduction_ = self._labels_from_distributions(
            self.label_distributions_, self.classes_, self.abstained_
        )

    def fit(self, X: Any, y: Any, *, k: int | None = None) -> "KNNLDPClassifier":
        """
        Build the k_max-nearest-neighbor structure once (if needed) and
        propagate labels for `k`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense feature matrix used only when the internal `KNNLDP` engine
            does not exist yet. After the first fit, subsequent calls reuse
            the cached structure and do not recompute distances.
        y : array-like of shape (n_samples,)
            Semi-supervised target with `-1` for unlabeled samples, following
            `sklearn.semi_supervised.LabelPropagation`'s convention.
        k : int or None, keyword-only, default=None
            Neighborhood size to use for this propagation. If None, `k_max`
            is used. Always re-validated and re-applied, even when the
            cached engine is reused.

        Returns
        -------
        KNNLDPClassifier
            The fitted estimator itself.
        """
        if not self._has_engine():
            X_checked = self._validate_X(X)
            k_max = self._validate_k_max(X_checked.shape[0])
            self.knn_ldp_ = KNNLDP(**self._engine_kwargs())
            self.knn_ldp_.fit(X_checked, k_max=k_max)
            self.k_max_ = k_max
            self._n_samples_fit_ = X_checked.shape[0]

        y_checked = self._validate_y(y, n_samples=self.knn_ldp_.n_samples_)
        k_value = self._validate_k(k, k_max=self.k_max_)
        self.knn_ldp_.propagate(y_checked, k=k_value)
        self.k_ = k_value
        self._sync_current_outputs()
        return self

    def fit_predict(self, X: Any, y: Any, *, k: int | None = None) -> np.ndarray:
        """
        Fit the estimator and return `predict(X)` (equal to `transduction_`).
        """
        return self.fit(X, y, k=k).predict(X)

    def _is_training_X(self, X: Any) -> bool:
        X_arr = np.asarray(X)
        return X_arr.shape == (self._n_samples_fit_, self.n_features_in_)

    def _check_predict_X(self, X: Any) -> None:
        if X is not None and not self._is_training_X(X):
            raise NotImplementedError(
                "KNNLDPClassifier only supports transductive prediction on "
                "the training set in this version (X=None or X equal in "
                "shape to the data passed to fit(...)). Inductive "
                "prediction on genuinely new query points (paper Section "
                "3.5) is not implemented yet."
            )

    def predict_proba(self, X: Any = None) -> np.ndarray:
        """
        Return the label probability distribution for the training set.

        Parameters
        ----------
        X : None or array-like, default=None
            If None, or if `X` has the same shape as the data used in
            `fit(...)`, the cached transductive result is returned with no
            recomputation. A genuinely new `X` is not supported yet.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Raw (non-renormalized) probability mass over `classes_`. Rows
            sum to less than 1 for partially or fully abstained instances;
            this is intentional (see class docstring) and matches the
            original paper's abstention semantics.
        """
        check_is_fitted(self, attributes=["knn_ldp_"])
        self._check_predict_X(X)
        return self.label_distributions_

    def predict(self, X: Any = None) -> np.ndarray:
        """
        Return crisp predictions for the training set.

        See `predict_proba` for the supported `X` values. Instances that
        abstained during propagation (no reachable labeled instance) are
        predicted as `-1`, reusing the same sentinel used for unlabeled
        input, so standard `sklearn.metrics` functions score them as errors
        with no extra handling required.
        """
        check_is_fitted(self, attributes=["knn_ldp_"])
        self._check_predict_X(X)
        return self.transduction_

    def get_fitted_engine(self) -> KNNLDP:
        """
        Return the fitted native KNNLDP engine.
        """
        check_is_fitted(self, attributes=["knn_ldp_"])
        return self.knn_ldp_
