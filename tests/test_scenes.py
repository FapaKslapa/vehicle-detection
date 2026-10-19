import numpy as np
import pytest

from vehicle_detection.data.scenes import (
    SceneConfig,
    assign_locations,
    assign_splits,
    cluster_scenes,
    leakage_report,
    random_split_crossing_share,
    rotation_invariant_similarity,
)


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def test_similarity_is_one_between_an_image_and_its_rotated_copy():
    base = unit(np.eye(4)[:3])
    rotated = np.zeros((3, 2, 4))
    rotated[:, 0] = base
    rotated[:, 1] = base
    rotated[1, 1] = base[0]
    similarity = rotation_invariant_similarity(base, rotated)
    assert similarity[0, 1] == pytest.approx(1.0)
    assert similarity[1, 0] == pytest.approx(1.0)
    assert similarity[0, 2] == pytest.approx(0.0)


def test_similarity_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(0)
    base = unit(rng.normal(size=(12, 8)))
    rotated = unit(rng.normal(size=(12, 3, 8)))
    similarity = rotation_invariant_similarity(base, rotated)
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 1.0)
    assert similarity.max() <= 1.0 + 1e-9


def test_similarity_without_rotations_is_plain_cosine():
    rng = np.random.default_rng(1)
    base = unit(rng.normal(size=(6, 5)))
    similarity = rotation_invariant_similarity(base, np.zeros((6, 0, 5)))
    expected = base @ base.T
    np.fill_diagonal(expected, 1.0)
    assert np.allclose(similarity, expected)


def blocks(sizes, intra=0.95, inter=0.2, seed=0):
    rng = np.random.default_rng(seed)
    total = sum(sizes)
    similarity = np.full((total, total), inter)
    start = 0
    for size in sizes:
        similarity[start : start + size, start : start + size] = intra
        start += size
    noise = rng.normal(scale=0.01, size=(total, total))
    similarity = similarity + (noise + noise.T) / 2
    np.fill_diagonal(similarity, 1.0)
    return np.clip(similarity, -1.0, 1.0)


def test_cluster_scenes_recovers_separated_blocks():
    similarity = blocks([25, 25, 25, 25])
    config = SceneConfig(neighbors=8, min_group_size=5, max_group_share=0.5)
    labels = cluster_scenes(similarity, config)
    assert len(set(labels)) == 4
    for start in range(0, 100, 25):
        assert len(set(labels[start : start + 25])) == 1


def test_cluster_scenes_is_deterministic():
    similarity = blocks([20, 30, 25], seed=3)
    config = SceneConfig(neighbors=6, min_group_size=5, max_group_share=0.6, seed=4)
    assert np.array_equal(cluster_scenes(similarity, config), cluster_scenes(similarity, config))


def test_cluster_scenes_merges_groups_below_the_minimum_size():
    similarity = blocks([30, 30, 2])
    config = SceneConfig(neighbors=6, min_group_size=5, max_group_share=0.6)
    labels = cluster_scenes(similarity, config)
    sizes = np.bincount(labels)
    assert sizes.min() >= 5


def test_cluster_scenes_splits_oversized_groups():
    big = blocks([30, 30, 30, 30], intra=0.9, inter=0.7, seed=5)
    total = 120 + 60
    similarity = np.full((total, total), 0.1)
    similarity[:120, :120] = big
    similarity[120:150, 120:150] = 0.95
    similarity[150:, 150:] = 0.95
    np.fill_diagonal(similarity, 1.0)
    config = SceneConfig(neighbors=8, min_group_size=5, max_group_share=0.4)
    labels = cluster_scenes(similarity, config)
    assert np.bincount(labels).max() <= 0.4 * total


def test_cluster_scenes_labels_are_contiguous():
    labels = cluster_scenes(
        blocks([20, 20, 20]), SceneConfig(neighbors=6, min_group_size=5, max_group_share=0.5)
    )
    assert sorted(set(labels)) == list(range(len(set(labels))))


def test_cluster_scenes_rejects_a_non_square_matrix():
    with pytest.raises(ValueError, match="quadrata"):
        cluster_scenes(np.zeros((3, 4)), SceneConfig())


def test_assign_locations_separates_two_clouds():
    rng = np.random.default_rng(0)
    left = rng.normal(loc=-5, size=(10, 4))
    right = rng.normal(loc=5, size=(10, 4))
    embeddings = np.vstack([left, right])
    labels = np.repeat(np.arange(20), 3)
    locations = assign_locations(np.repeat(embeddings, 3, axis=0), labels, seed=0)
    assert len({locations[i] for i in range(10)}) == 1
    assert len({locations[i] for i in range(10, 20)}) == 1
    assert locations[0] != locations[10]
    assert set(locations.values()) == {"loc0", "loc1"}


def test_assign_locations_with_one_group_is_a_single_location():
    embeddings = np.ones((4, 3))
    assert assign_locations(embeddings, np.zeros(4, dtype=int), seed=0) == {0: "loc0"}


def make_groups(seed=0, count=40):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(8, 60, size=count)
    labels = np.repeat(np.arange(count), sizes)
    counts = np.zeros((len(labels), 4))
    counts[:, 0] = rng.poisson(3, len(labels))
    counts[:, 1] = rng.poisson(1.5, len(labels))
    rare_groups = rng.choice(count, size=8, replace=False)
    for group in rare_groups:
        counts[np.flatnonzero(labels == group), 3] = rng.poisson(1.0, sizes[group])
    locations = {group: f"loc{group % 2}" for group in range(count)}
    return labels, locations, counts


def split_of_images(labels, assignment):
    return np.array([assignment[label] for label in labels])


def test_assign_splits_covers_every_group_with_valid_names():
    labels, locations, counts = make_groups()
    assignment = assign_splits(labels, locations, counts, SceneConfig())
    assert set(assignment) == set(labels)
    assert set(assignment.values()) == {"train", "val", "test"}


def test_assign_splits_keeps_image_proportions_close_to_targets():
    labels, locations, counts = make_groups()
    splits = split_of_images(labels, assign_splits(labels, locations, counts, SceneConfig()))
    shares = {name: (splits == name).mean() for name in ("train", "val", "test")}
    assert shares["train"] == pytest.approx(0.70, abs=0.04)
    assert shares["val"] == pytest.approx(0.15, abs=0.04)
    assert shares["test"] == pytest.approx(0.15, abs=0.04)


def test_assign_splits_balances_locations_in_val_and_test():
    labels, locations, counts = make_groups()
    splits = split_of_images(labels, assign_splits(labels, locations, counts, SceneConfig()))
    location_of_image = np.array([locations[label] for label in labels])
    overall = (location_of_image == "loc1").mean()
    for name in ("train", "val", "test"):
        share = (location_of_image[splits == name] == "loc1").mean()
        assert share == pytest.approx(overall, abs=0.15)


def test_assign_splits_puts_rare_classes_in_val_and_test():
    labels, locations, counts = make_groups()
    splits = split_of_images(labels, assign_splits(labels, locations, counts, SceneConfig()))
    for name in ("train", "val", "test"):
        assert counts[splits == name, 3].sum() > 0


def test_assign_splits_is_deterministic_and_seed_sensitive():
    labels, locations, counts = make_groups()
    first = assign_splits(labels, locations, counts, SceneConfig(seed=1))
    assert first == assign_splits(labels, locations, counts, SceneConfig(seed=1))


def test_assign_splits_needs_at_least_three_groups():
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="almeno tre gruppi"):
        assign_splits(labels, {0: "loc0", 1: "loc0"}, np.ones((4, 2)), SceneConfig())


def test_scene_config_rejects_ratios_that_leave_no_room_for_test():
    with pytest.raises(ValueError):
        SceneConfig(train=0.9, val=0.2)


def test_leakage_report_counts_near_duplicate_pairs_across_splits():
    similarity = np.full((6, 6), 0.1)
    for a, b in ((0, 1), (2, 3), (4, 5), (1, 2)):
        similarity[a, b] = similarity[b, a] = 0.99
    np.fill_diagonal(similarity, 1.0)
    splits = np.array(["train", "train", "val", "val", "test", "test"])
    report = leakage_report(similarity, splits, threshold=0.95)
    assert report.near_duplicate_pairs == 4
    assert report.crossing_pairs == 1
    assert report.crossing_share == pytest.approx(0.25)


def test_leakage_report_with_no_near_duplicates_is_zero():
    similarity = np.full((4, 4), 0.1)
    np.fill_diagonal(similarity, 1.0)
    report = leakage_report(similarity, np.array(["train"] * 4), threshold=0.95)
    assert report.near_duplicate_pairs == 0
    assert report.crossing_share == 0.0


def test_random_split_crossing_share_matches_the_closed_form():
    assert random_split_crossing_share(0.7, 0.15) == pytest.approx(1 - (0.49 + 0.0225 + 0.0225))


def test_lower_resolution_gives_fewer_larger_scenes():
    similarity = blocks([30, 30, 30, 30], intra=0.95, inter=0.85, seed=7)
    coarse = SceneConfig(neighbors=40, min_group_size=5, max_group_share=1.0, resolution=0.05)
    fine = SceneConfig(neighbors=40, min_group_size=5, max_group_share=1.0, resolution=4.0)
    assert len(set(cluster_scenes(similarity, coarse))) < len(set(cluster_scenes(similarity, fine)))


def test_scene_config_rejects_a_non_positive_resolution():
    with pytest.raises(ValueError):
        SceneConfig(resolution=0.0)
