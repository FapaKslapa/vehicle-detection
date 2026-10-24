"""How much of the class error is only fine-grained car types being mixed up.

TRANSSET labels four passenger-car types (car, hatchback, sedan, suv) that are hard to tell
apart from directly above, and "car" may overlap the other three as a label. Merging them
into one "passenger car" class and recomputing class accuracy, AP and per-class counting
error shows how much better the system is at the coarser question.

Merging the confusion matrix is exact: the matrix pairs boxes regardless of class, so the
pairs do not change, only which cells they fall in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from vehicle_detection.evaluation.errors import NO_BOX, category_names, localisation_summary

DEFAULT_GROUP_TEXT = "passenger car=car,hatchback,sedan,suv"
DISABLED = "none"


@dataclass(frozen=True, slots=True)
class ClassGroup:
    name: str
    members: tuple[str, ...]

    def present(self, names: Sequence[str]) -> tuple[str, ...]:
        return tuple(member for member in self.members if member in names)

    def applies_to(self, names: Sequence[str]) -> bool:
        return len(self.present(names)) >= 2


def parse_group(text: str | None) -> ClassGroup | None:
    """``"passenger car=car,hatchback,sedan,suv"`` -> ClassGroup; ``"none"`` -> None."""
    if text is None or text.strip().lower() == DISABLED:
        return None
    name, separator, members = text.partition("=")
    parts = tuple(member.strip() for member in members.split(",") if member.strip())
    if not separator or not name.strip() or len(parts) < 2:
        raise ValueError(f"expected 'name=member,member,...' with at least two members: {text!r}")
    if len(set(parts)) != len(parts):
        raise ValueError(f"a class is listed twice in the group: {text!r}")
    return ClassGroup(name.strip(), parts)


def merge_categories(
    gt: Mapping[str, Any], results: Sequence[Mapping[str, Any]], group: ClassGroup
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ground truth and results with the group's classes relabelled as one category.

    The merged category takes the smallest id of its members; the other classes keep theirs.
    """
    names = category_names(gt)
    members = group.present(list(names.values()))
    if len(members) < 2:
        raise ValueError(f"fewer than two of {group.members} are categories here")
    if group.name in names.values() and group.name not in members:
        raise ValueError(f"'{group.name}' is already the name of another category")
    member_ids = sorted(category_id for category_id, name in names.items() if name in members)
    merged_id = member_ids[0]
    categories = [
        {"id": category_id, "name": name}
        for category_id, name in names.items()
        if category_id not in member_ids
    ]
    categories.append({"id": merged_id, "name": group.name})
    categories.sort(key=lambda category: category["id"])

    def relabel(item: Mapping[str, Any]) -> dict[str, Any]:
        category_id = merged_id if item["category_id"] in member_ids else item["category_id"]
        return {**item, "category_id": category_id}

    return (
        {
            **gt,
            "categories": categories,
            "annotations": [relabel(annotation) for annotation in gt["annotations"]],
        },
        [relabel(result) for result in results],
    )


def merge_matrix(matrix: pd.DataFrame, group: ClassGroup) -> pd.DataFrame:
    """Add up the group's rows and columns of a confusion matrix into one."""
    members = list(group.present(list(matrix.index)))
    if len(members) < 2:
        raise ValueError(f"fewer than two of {group.members} are in the matrix")
    first = min(list(matrix.index).index(member) for member in members)
    labels = [label for label in matrix.index if label not in members]
    labels.insert(first, group.name)

    def merged_label(label: str) -> str:
        return group.name if label in members else label

    merged = matrix.groupby(merged_label).sum()
    merged = merged.T.groupby(merged_label).sum().T
    merged = merged.reindex(index=labels, columns=labels)
    merged.index.name = matrix.index.name
    merged.columns.name = matrix.columns.name
    return merged


def merge_summary(matrix: pd.DataFrame, group: ClassGroup) -> dict[str, Any]:
    members = list(group.present(list(matrix.index)))
    real = [label for label in matrix.index if label != NO_BOX]
    found = int(matrix.loc[real, real].to_numpy().sum())
    correct = int(sum(matrix.loc[label, label] for label in real))
    wrong = found - correct
    within = int(
        sum(
            matrix.loc[true, predicted]
            for true in members
            for predicted in members
            if true != predicted
        )
    )
    group_found = int(matrix.loc[members, real].to_numpy().sum())
    group_named_in_group = int(matrix.loc[members, members].to_numpy().sum())
    group_exact = int(sum(matrix.loc[member, member] for member in members))
    merged = localisation_summary(merge_matrix(matrix, group))
    return {
        "group": group.name,
        "members": ", ".join(members),
        "class_accuracy_when_found": _ratio(correct, found),
        "class_accuracy_merged": merged.class_accuracy,
        "class_errors": wrong,
        "errors_within_group": within,
        "share_of_class_errors_within_group": _ratio(within, wrong),
        "group_vehicles_found": group_found,
        "group_exact_type_accuracy": _ratio(group_exact, group_found),
        "group_any_member_accuracy": _ratio(group_named_in_group, group_found),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")
