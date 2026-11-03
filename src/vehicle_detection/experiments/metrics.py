"""Moved to ``vehicle_detection.evaluation.metrics``.

This module only re-exports it, so code on branches that still import
``vehicle_detection.experiments.metrics`` keeps working. New code should import from
``vehicle_detection.evaluation.metrics``.
"""

from vehicle_detection.evaluation.metrics import (
    AP,
    AP50,
    AP_LARGE,
    AP_MEDIUM,
    AP_SMALL,
    AR_LARGE,
    AR_MEDIUM,
    AR_SMALL,
    MATCH_IOU,
    MAX_DETECTIONS,
    NOT_DEFINED,
    CountingMetrics,
    DetectionMetrics,
    MatchMetrics,
    PrecisionRecall,
    best_threshold,
    counting_metrics,
    detection_metrics,
    match_counts,
    match_metrics,
)

__all__ = [
    "AP",
    "AP50",
    "AP_LARGE",
    "AP_MEDIUM",
    "AP_SMALL",
    "AR_LARGE",
    "AR_MEDIUM",
    "AR_SMALL",
    "MATCH_IOU",
    "MAX_DETECTIONS",
    "NOT_DEFINED",
    "CountingMetrics",
    "DetectionMetrics",
    "MatchMetrics",
    "PrecisionRecall",
    "best_threshold",
    "counting_metrics",
    "detection_metrics",
    "match_counts",
    "match_metrics",
]
