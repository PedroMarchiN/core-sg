from __future__ import annotations

import heapq
from itertools import count
from time import time
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from sklearn.utils.validation import validate_data
except ImportError:
    validate_data = None

from .core_sg import CoreSG
from .knn import knn_from_precomputed

ProgressCallback = Callable[[str, float, dict[str, Any]], None]

NEIGHBOR_SPACES = ("euclidean", "core_sg", "mutual_reachability_exact")


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _undirected_keys(u: np.ndarray, v: np.ndarray, n_nodes: int) -> np.ndarray:
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    return np.minimum(u, v) * np.int64(n_nodes) + np.maximum(u, v)


def _emit_progress(
    event: str,
    elapsed: float,
    *,
    verbose: int = 0,
    progress_callback: ProgressCallback | None = None,
    message: str | None = None,
    **info: Any,
) -> None:
    if progress_callback is not None:
        progress_callback(event, elapsed, dict(info))
    if verbose:
        print(message or f"{event} done in {elapsed:.2f}s")


def build_knn_ldp_graph_from_data(
    X: np.ndarray,
    k_max: int,
    *,
    metric: str = "euclidean",
    p: int = 2,
    pairwise_dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray]:
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


def dedupe_core_sg_edges(
    edges: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1] < 3:
        raise ValueError("edges must have shape (n_edges, 3).")

    u = edges[:, 0].astype(np.int64, copy=False)
    v = edges[:, 1].astype(np.int64, copy=False)
    w = edges[:, 2].astype(np.float64, copy=False)

    key = _undirected_keys(u, v, n_nodes)
    _, first = np.unique(key, return_index=True)
    return u[first], v[first], w[first]


def build_neighbor_graph_from_edges(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    n_nodes: int,
    k_max: int,
    *,
    mst_keys: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if k_max <= 0:
        raise ValueError("k_max must be >= 1.")

    src = np.concatenate([u, v]).astype(np.int64, copy=False)
    dst = np.concatenate([v, u]).astype(np.int64, copy=False)
    weight = np.concatenate([w, w]).astype(np.float64, copy=False)

    if mst_keys is None:
        order = np.lexsort((weight, src))
    else:
        key = _undirected_keys(src, dst, n_nodes)
        pos = np.searchsorted(mst_keys, key)
        pos_clipped = np.minimum(pos, max(mst_keys.size - 1, 0))
        is_mst = (mst_keys.size > 0) & (pos < mst_keys.size)
        is_mst = is_mst & (mst_keys[pos_clipped] == key)
        group = np.where(is_mst, 0, 1)
        within = np.where(is_mst, -weight, weight)
        order = np.lexsort((within, group, src))

    src_sorted = src[order]
    dst_sorted = dst[order]
    weight_sorted = weight[order]

    indptr = np.searchsorted(src_sorted, np.arange(n_nodes + 1, dtype=np.int64))
    degree = np.diff(indptr)
    if degree.min() < k_max:
        raise ValueError(
            f"Point {int(np.argmin(degree))} has degree {int(degree.min())} "
            f"< k_max={k_max}; the edge set is not a valid Core-SG support."
        )

    cols = indptr[:n_nodes, None] + np.arange(k_max, dtype=np.int64)[None, :]
    return dst_sorted[cols], weight_sorted[cols]


def mutual_reachability_knn(
    D: np.ndarray, core_k_list: np.ndarray, min_pts: int, k_max: int
) -> tuple[np.ndarray, np.ndarray]:
    D = np.asarray(D, dtype=np.float64)
    core = np.ascontiguousarray(core_k_list[:, min_pts - 1], dtype=np.float64)

    mreach = np.maximum(D, core[:, None])
    mreach = np.maximum(mreach, core[None, :])
    return knn_from_precomputed(mreach, k=k_max, include_self=False)


def build_reverse_knn_index(
    idxs_graph: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
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


class _MaxPriorityQueue:

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


def propagate_knn_ldp(
    idxs_graph: np.ndarray,
    y_encoded: np.ndarray,
    n_classes: int,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
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
            abstain_idx = np.array(pq.remaining_nodes() + [node], dtype=np.int64)
            label_dist[abstain_idx, :] = 0.0
            label_dist[abstain_idx, unknown_col] = 1.0
            abstained[abstain_idx] = True
            resolved[abstain_idx] = True
            break

    return label_dist, abstained


class KNNLDP:

    def __init__(
        self,
        metric: str = "euclidean",
        p: int = 2,
        verbose: int = 0,
        progress_callback: ProgressCallback | None = None,
        neighbor_space: str = "euclidean",
        min_pts: int | None = None,
        force_mst_edges: bool = False,
    ) -> None:
        if neighbor_space not in NEIGHBOR_SPACES:
            raise ValueError(
                f"neighbor_space must be one of {NEIGHBOR_SPACES}, got "
                f"{neighbor_space!r}."
            )

        self.metric = metric
        self.p = p
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.neighbor_space = neighbor_space
        self.min_pts = min_pts
        self.force_mst_edges = force_mst_edges

        self.n_samples_ = None
        self.k_max_ = None
        self.min_pts_ = None
        self.idxs_graph_ = None
        self.dists_graph_ = None
        self.core_sg_ = None

        self.k_ = None
        self.classes_ = None
        self.label_distributions_extended_ = None
        self.abstained_ = None

    def _resolve_min_pts(self, k_max: int) -> int:
        min_pts = k_max if self.min_pts is None else self.min_pts
        if not _is_integer(min_pts):
            raise ValueError("min_pts must be an integer or None.")

        min_pts = int(min_pts)
        if min_pts < 2:
            raise ValueError("min_pts must be >= 2 (Core-SG core-distance bound).")
        if min_pts > k_max:
            raise ValueError("min_pts must satisfy 2 <= min_pts <= k_max.")
        return min_pts

    def _graph_from_cached_core_sg(
        self, k_max: int, min_pts: int
    ) -> tuple[np.ndarray, np.ndarray]:
        core_sg = self.core_sg_
        n = core_sg.n_samples_

        if self.neighbor_space == "mutual_reachability_exact":
            return mutual_reachability_knn(
                core_sg.distance_matrix_, core_sg.core_distances_, min_pts, k_max
            )

        edges = core_sg.get_core_sg_mutual_reachability_distance(min_pts)
        u, v, w = dedupe_core_sg_edges(edges, n)

        mst_keys = None
        if self.force_mst_edges:
            mst = np.asarray(core_sg.extract_mst_from_core_sg(min_pts))
            mst_keys = np.sort(
                _undirected_keys(
                    mst[:, 0].astype(np.int64), mst[:, 1].astype(np.int64), n
                )
            )

        return build_neighbor_graph_from_edges(u, v, w, n, k_max, mst_keys=mst_keys)

    def _build_core_sg_backed_graph(
        self, X: np.ndarray, k_max: int
    ) -> tuple[np.ndarray, np.ndarray]:
        min_pts = self._resolve_min_pts(k_max)

        core_sg = CoreSG(verbose=self.verbose)
        core_sg.fit(X, k_max)
        self.core_sg_ = core_sg
        self.min_pts_ = min_pts

        idxs_graph, dists_graph = self._graph_from_cached_core_sg(k_max, min_pts)

        if self.neighbor_space == "core_sg":
            core_sg.distance_matrix_ = None
            core_sg._tree_to_labels_data_ = None

        return idxs_graph, dists_graph

    def set_min_pts(self, min_pts: int) -> "KNNLDP":
        self._ensure_fitted()
        if self.neighbor_space == "euclidean":
            raise ValueError(
                "min_pts does not apply to neighbor_space='euclidean'; "
                "Euclidean neighbors do not depend on core distances."
            )

        resolved = int(min_pts)
        if not _is_integer(min_pts):
            raise ValueError("min_pts must be an integer.")
        if resolved < 2:
            raise ValueError("min_pts must be >= 2 (Core-SG core-distance bound).")
        if resolved > self.k_max_:
            raise ValueError("min_pts must satisfy 2 <= min_pts <= k_max.")

        t0 = time()
        self.idxs_graph_, self.dists_graph_ = self._graph_from_cached_core_sg(
            self.k_max_, resolved
        )
        t1 = time()
        _emit_progress(
            "reweight",
            t1 - t0,
            verbose=self.verbose,
            progress_callback=self.progress_callback,
            message=f"kNN-LDP min_pts = {resolved} reweight done in {t1 - t0:.2f}s",
            min_pts=resolved,
        )

        self.min_pts_ = resolved
        return self

    def fit(self, X: np.ndarray, k_max: int) -> "KNNLDP":
        X = np.asarray(X)

        t0 = time()
        if self.neighbor_space == "euclidean":
            idxs_graph, dists_graph = build_knn_ldp_graph_from_data(
                X, k_max, metric=self.metric, p=self.p
            )
        else:
            idxs_graph, dists_graph = self._build_core_sg_backed_graph(X, k_max)
        t1 = time()
        _emit_progress(
            "build",
            t1 - t0,
            verbose=self.verbose,
            progress_callback=self.progress_callback,
            message=f"kNN-LDP graph build done in {t1 - t0:.2f}s",
            k_max=k_max,
            neighbor_space=self.neighbor_space,
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


class KNNLDPClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        k_max: int,
        metric: str = "euclidean",
        p: int = 2,
        verbose: int = 0,
        progress_callback: ProgressCallback | None = None,
        neighbor_space: str = "euclidean",
        min_pts: int | None = None,
        force_mst_edges: bool = False,
    ) -> None:
        self.k_max = k_max
        self.metric = metric
        self.p = p
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.neighbor_space = neighbor_space
        self.min_pts = min_pts
        self.force_mst_edges = force_mst_edges

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
        if self.neighbor_space != "euclidean" and k_max < 2:
            raise ValueError(
                f"k_max must be >= 2 for neighbor_space={self.neighbor_space!r} "
                "(Core-SG requires it to build core distances)."
            )
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
            "neighbor_space": self.neighbor_space,
            "min_pts": self.min_pts,
            "force_mst_edges": self.force_mst_edges,
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

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        k: int | None = None,
        min_pts: int | None = None,
    ) -> "KNNLDPClassifier":
        if not self._has_engine():
            X_checked = self._validate_X(X)
            k_max = self._validate_k_max(X_checked.shape[0])
            self.knn_ldp_ = KNNLDP(**self._engine_kwargs())
            self.knn_ldp_.fit(X_checked, k_max=k_max)
            self.k_max_ = k_max
            self._n_samples_fit_ = X_checked.shape[0]

        if min_pts is not None and min_pts != self.knn_ldp_.min_pts_:
            self.knn_ldp_.set_min_pts(min_pts)

        y_checked = self._validate_y(y, n_samples=self.knn_ldp_.n_samples_)
        k_value = self._validate_k(k, k_max=self.k_max_)
        self.knn_ldp_.propagate(y_checked, k=k_value)
        self.k_ = k_value
        self.min_pts_ = self.knn_ldp_.min_pts_
        self._sync_current_outputs()
        return self

    def fit_predict(
        self,
        X: Any,
        y: Any,
        *,
        k: int | None = None,
        min_pts: int | None = None,
    ) -> np.ndarray:
        return self.fit(X, y, k=k, min_pts=min_pts).predict(X)

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
        check_is_fitted(self, attributes=["knn_ldp_"])
        self._check_predict_X(X)
        return self.label_distributions_

    def predict(self, X: Any = None) -> np.ndarray:
        check_is_fitted(self, attributes=["knn_ldp_"])
        self._check_predict_X(X)
        return self.transduction_

    def get_fitted_engine(self) -> KNNLDP:
        check_is_fitted(self, attributes=["knn_ldp_"])
        return self.knn_ldp_
