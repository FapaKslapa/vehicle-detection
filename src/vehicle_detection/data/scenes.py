from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SPLITS = ("train", "val", "test")
LOCATION_COUNT = 2
KMEANS_ITERATIONS = 30
MAX_RESOLUTION_STEPS = 8
RESOLUTION_GROWTH = 2.0
ANNEALING_RESTARTS = 4
ANNEALING_STEPS = 10000
START_TEMPERATURE = 0.05
END_TEMPERATURE = 0.0002
IMAGE_WEIGHT = 4.0
LOCATION_WEIGHT = 4.0
CLASS_WEIGHT = 1.0
EPSILON = 1e-12


class SceneConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    neighbors: int = Field(default=15, ge=2)
    resolution: float = Field(default=0.8, gt=0.0)
    min_group_size: int = Field(default=8, ge=1)
    max_group_share: float = Field(default=0.25, gt=0.0, le=1.0)
    train: float = Field(default=0.7, gt=0, lt=1)
    val: float = Field(default=0.15, gt=0, lt=1)
    near_duplicate_similarity: float = Field(default=0.95, gt=0.0, le=1.0)
    seed: int = 0

    @model_validator(mode="after")
    def _ratios_leave_room_for_test(self) -> SceneConfig:
        if self.train + self.val >= 1:
            raise ValueError("train + val deve essere inferiore a 1")
        return self

    @property
    def test(self) -> float:
        return 1.0 - self.train - self.val


@dataclass(frozen=True, slots=True)
class LeakageReport:
    near_duplicate_pairs: int
    crossing_pairs: int
    crossing_share: float


def rotation_invariant_similarity(base: np.ndarray, rotated: np.ndarray) -> np.ndarray:
    base = _normalise(np.asarray(base, dtype=np.float64))
    rotated = _normalise(np.asarray(rotated, dtype=np.float64))
    similarity = base @ base.T
    for angle in range(rotated.shape[1]):
        similarity = np.maximum(similarity, base @ rotated[:, angle].T)
    similarity = np.maximum(similarity, similarity.T)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def cluster_scenes(similarity: np.ndarray, config: SceneConfig) -> np.ndarray:
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(f"matrice di similarità non quadrata: shape={similarity.shape}")
    count = similarity.shape[0]
    labels = np.zeros(count, dtype=int)
    if count == 0:
        return labels
    cap = max(1, int(config.max_group_share * count))
    first_pass = [
        np.array(part, dtype=int) for part in _communities(similarity, config, config.resolution)
    ]
    groups = _split_oversized(first_pass, similarity, config, cap)
    groups = _merge_small(groups, similarity, config.min_group_size)
    for label, members in enumerate(sorted(groups, key=lambda members: int(members.min()))):
        labels[members] = label
    return labels


def assign_locations(embeddings: np.ndarray, labels: np.ndarray, seed: int) -> dict[int, str]:
    groups = sorted(set(int(label) for label in labels))
    if len(groups) < LOCATION_COUNT:
        return dict.fromkeys(groups, "loc0")
    embeddings = _normalise(np.asarray(embeddings, dtype=np.float64))
    means = np.stack([embeddings[labels == group].mean(axis=0) for group in groups])
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(means)))
    second = int(np.argmax(np.linalg.norm(means - means[first], axis=1)))
    centers = means[[first, second]].copy()
    assignment = np.zeros(len(means), dtype=int)
    for _ in range(KMEANS_ITERATIONS):
        distances = np.linalg.norm(means[:, None, :] - centers[None, :, :], axis=2)
        assignment = distances.argmin(axis=1)
        for cluster in range(LOCATION_COUNT):
            if (assignment == cluster).any():
                centers[cluster] = means[assignment == cluster].mean(axis=0)
    first_cluster = assignment[0]
    return {
        group: "loc0" if assignment[index] == first_cluster else "loc1"
        for index, group in enumerate(groups)
    }


def assign_splits(
    labels: np.ndarray,
    locations: Mapping[int, str],
    class_counts: np.ndarray,
    config: SceneConfig,
) -> dict[int, str]:
    labels = np.asarray(labels)
    groups = sorted(set(int(label) for label in labels))
    if len(groups) < len(SPLITS):
        raise ValueError(
            f"servono almeno tre gruppi per costruire train, val e test: trovati {len(groups)}"
        )
    features, weights = _group_features(labels, groups, locations, np.asarray(class_counts))
    targets = np.array([config.train, config.val, config.test])
    best_loss, best_assignment = np.inf, None
    for restart in range(ANNEALING_RESTARTS):
        rng = np.random.default_rng([config.seed, restart])
        assignment, loss = _anneal(features, weights, targets, rng)
        if loss < best_loss:
            best_loss, best_assignment = loss, assignment
    assert best_assignment is not None
    return {group: SPLITS[best_assignment[index]] for index, group in enumerate(groups)}


def leakage_report(similarity: np.ndarray, splits: np.ndarray, threshold: float) -> LeakageReport:
    splits = np.asarray(splits)
    upper = np.triu(np.asarray(similarity) >= threshold, k=1)
    rows, cols = np.nonzero(upper)
    near_duplicates = len(rows)
    crossing = int((splits[rows] != splits[cols]).sum())
    share = crossing / near_duplicates if near_duplicates else 0.0
    return LeakageReport(near_duplicates, crossing, share)


def random_split_crossing_share(train: float, val: float) -> float:
    test = 1.0 - train - val
    return 1.0 - (train**2 + val**2 + test**2)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, EPSILON)


def _knn_graph(similarity: np.ndarray, neighbors: int) -> nx.Graph:
    count = similarity.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    if count < 2:
        return graph
    masked = similarity.copy()
    np.fill_diagonal(masked, -np.inf)
    width = min(neighbors, count - 1)
    nearest = np.argpartition(-masked, width - 1, axis=1)[:, :width]
    for node in range(count):
        for other in nearest[node]:
            weight = max(float(similarity[node, other]), EPSILON)
            if not graph.has_edge(node, int(other)) or graph[node][int(other)]["weight"] < weight:
                graph.add_edge(node, int(other), weight=weight)
    return graph


def _communities(
    similarity: np.ndarray, config: SceneConfig, resolution: float
) -> list[np.ndarray]:
    graph = _knn_graph(similarity, config.neighbors)
    found = louvain_communities(graph, weight="weight", resolution=resolution, seed=config.seed)
    return [np.array(sorted(community), dtype=int) for community in found]


def _split_oversized(
    groups: list[np.ndarray], similarity: np.ndarray, config: SceneConfig, cap: int
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    pending = list(groups)
    while pending:
        members = pending.pop()
        if len(members) <= cap or len(members) <= 1:
            result.append(members)
            continue
        parts = _try_split(members, similarity, config)
        if len(parts) == 1:
            result.append(members)
            continue
        pending.extend(parts)
    return result


def _try_split(
    members: np.ndarray, similarity: np.ndarray, config: SceneConfig
) -> list[np.ndarray]:
    sub = similarity[np.ix_(members, members)]
    resolution = max(config.resolution, 1.0)
    for _ in range(MAX_RESOLUTION_STEPS):
        parts = _communities(sub, config, resolution)
        if len(parts) > 1:
            return [members[part] for part in parts]
        resolution *= RESOLUTION_GROWTH
    if len(members) < 2:
        return [members]
    distance = np.clip(1.0 - sub, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    cut = fcluster(tree, 2, criterion="maxclust")
    return [members[cut == value] for value in sorted(set(cut))]


def _merge_small(
    groups: list[np.ndarray], similarity: np.ndarray, minimum: int
) -> list[np.ndarray]:
    groups = [members for members in groups]
    while len(groups) > 1:
        sizes = [len(members) for members in groups]
        smallest = int(np.argmin(sizes))
        if sizes[smallest] >= minimum:
            break
        small = groups[smallest]
        scores = [
            float(similarity[np.ix_(small, other)].mean()) if index != smallest else -np.inf
            for index, other in enumerate(groups)
        ]
        target = int(np.argmax(scores))
        groups[target] = np.concatenate([groups[target], small])
        del groups[smallest]
    return groups


def _group_features(
    labels: np.ndarray,
    groups: list[int],
    locations: Mapping[int, str],
    class_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    location_names = sorted({locations[group] for group in groups})
    columns: list[np.ndarray] = []
    weights: list[float] = []
    sizes = np.array([(labels == group).sum() for group in groups], dtype=np.float64)
    columns.append(sizes)
    weights.append(IMAGE_WEIGHT)
    for name in location_names:
        columns.append(np.where([locations[group] == name for group in groups], sizes, 0.0))
        weights.append(LOCATION_WEIGHT)
    for klass in range(class_counts.shape[1]):
        per_group = np.array(
            [class_counts[labels == group, klass].sum() for group in groups], dtype=np.float64
        )
        if per_group.sum() > 0:
            columns.append(per_group)
            weights.append(CLASS_WEIGHT)
    return np.stack(columns, axis=1), np.array(weights)


def _anneal(
    features: np.ndarray, weights: np.ndarray, targets: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    group_count = features.shape[0]
    total = features.sum(axis=0)
    normalised = features / np.maximum(total, EPSILON)
    assignment = _greedy_start(normalised[:, 0], targets, rng)
    sums = np.stack([normalised[assignment == split].sum(axis=0) for split in range(len(SPLITS))])

    def row_loss(row: np.ndarray, split: int) -> float:
        return float((weights * ((row - targets[split]) / targets[split]) ** 2).sum())

    losses = np.array([row_loss(sums[split], split) for split in range(len(SPLITS))])
    current = float(losses.sum())
    best, best_assignment = current, assignment.copy()
    for step in range(ANNEALING_STEPS):
        temperature = START_TEMPERATURE * (END_TEMPERATURE / START_TEMPERATURE) ** (
            step / ANNEALING_STEPS
        )
        group = int(rng.integers(group_count))
        source = int(assignment[group])
        target = int(rng.integers(len(SPLITS)))
        if target == source or (assignment == source).sum() == 1:
            continue
        new_source = sums[source] - normalised[group]
        new_target = sums[target] + normalised[group]
        candidate = current - losses[source] - losses[target]
        candidate += row_loss(new_source, source) + row_loss(new_target, target)
        if candidate < current or rng.random() < np.exp((current - candidate) / temperature):
            sums[source], sums[target] = new_source, new_target
            losses[source] = row_loss(new_source, source)
            losses[target] = row_loss(new_target, target)
            assignment[group] = target
            current = float(losses.sum())
            if current < best:
                best, best_assignment = current, assignment.copy()
    return best_assignment, best


def _greedy_start(sizes: np.ndarray, targets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    assignment = np.zeros(len(sizes), dtype=int)
    filled = np.zeros(len(SPLITS))
    jitter = rng.random(len(sizes)) * 1e-9
    for group in np.argsort(-(sizes + jitter)):
        split = int(np.argmax((targets - filled) / targets))
        assignment[group] = split
        filled[split] += sizes[group]
    for split in range(len(SPLITS)):
        if not (assignment == split).any():
            donor = int(np.argmax(np.bincount(assignment, minlength=len(SPLITS))))
            assignment[int(np.flatnonzero(assignment == donor)[-1])] = split
    return assignment
