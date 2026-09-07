import numpy as np
import pytest
from coatopt.utils.metrics import (
    compute_target_yield,
    compute_objective_breakdown,
    compute_asf_scores,
    compute_roi_hypervolume,
    evaluate_dataset_proximity_metrics,
)


def create_sample_designs():
    # 3 sample designs
    # Design 1: Transmission 5 ppm, Abs 0.2 ppm, TN 3.5e-21 (meets all targets: T <= 10, Abs <= 0.3, TN <= 4.0e-21)
    # Design 2: Transmission 12 ppm, Abs 0.25 ppm, TN 3.8e-21 (T fails at alpha=0, but meets at alpha=0.25)
    # Design 3: Transmission 25 ppm, Abs 0.5 ppm, TN 5.0e-21 (fails all)
    designs = [
        {
            "transmission": 5.0,
            "reflectivity": 1.0 - 5.0e-6,
            "absorption": 0.20,
            "thermal_noise": 3.5e-21,
            "total_thickness": 4500.0,
        },
        {
            "transmission": 12.0,
            "reflectivity": 1.0 - 12.0e-6,
            "absorption": 0.25,
            "thermal_noise": 3.8e-21,
            "total_thickness": 5000.0,
        },
        {
            "transmission": 25.0,
            "reflectivity": 1.0 - 25.0e-6,
            "absorption": 0.50,
            "thermal_noise": 5.0e-21,
            "total_thickness": 5500.0,
        },
    ]
    return designs


def test_compute_target_yield_transmission():
    designs = create_sample_designs()
    targets = {
        "reflectivity": 0.999990,
        "transmission": 10.0,
        "absorption": 0.30,
        "thermal_noise": 4.0e-21,
    }
    res = compute_target_yield(
        designs,
        targets=targets,
        primary_metric="transmission",
    )
    # At alpha = 0: only Design 1 passes (1/3 = 33.33%)
    assert res["count_zero"] == 1
    assert pytest.approx(res["yield_zero"], 0.1) == 33.33

    # At alpha = 0.25 (tolerance 25%):
    # T thresh: 10 * 1.25 = 12.5 -> Design 1 (5) and Design 2 (12) pass
    # Abs thresh: 0.3 * 1.25 = 0.375 -> Design 1 (0.2) and Design 2 (0.25) pass
    # TN thresh: 4.0e-21 * 1.25 = 5.0e-21 -> Design 1 (3.5e-21) and Design 2 (3.8e-21) pass
    # Count should be 2 (66.67%)
    curve = {point["tolerance"]: point["count"] for point in res["yield_curve"]}
    assert curve[0.25] == 2


def test_compute_target_yield_reflectivity():
    designs = create_sample_designs()
    targets = {
        "reflectivity": 0.999990,  # 1-R = 10 ppm
        "absorption": 0.30,
        "thermal_noise": 4.0e-21,
    }
    res = compute_target_yield(
        designs,
        targets=targets,
        primary_metric="reflectivity",
    )
    assert res["count_zero"] == 1


def test_compute_objective_breakdown_transmission():
    designs = create_sample_designs()
    targets = {
        "reflectivity": 0.999990,
        "transmission": 10.0,
        "absorption": 0.30,
        "thermal_noise": 4.0e-21,
    }
    breakdown = compute_objective_breakdown(
        designs,
        targets=targets,
        primary_metric="transmission",
    )
    assert len(breakdown) == 3
    t_obj = breakdown[0]
    assert t_obj["objective"] == "transmission"
    assert t_obj["pass_count"] == 1
    assert pytest.approx(t_obj["pass_pct"], 0.1) == 33.33


def test_compute_asf_scores_transmission():
    designs = create_sample_designs()
    targets = {
        "reflectivity": 0.999990,
        "transmission": 10.0,
        "absorption": 0.30,
        "thermal_noise": 4.0e-21,
    }
    scores = compute_asf_scores(
        designs,
        targets=targets,
        primary_metric="transmission",
    )
    # Design 1 exceeds all targets so its score is negative
    assert scores[0] < 0
    assert np.argmin(scores) == 0


def test_compute_roi_hypervolume_transmission():
    designs = create_sample_designs()
    targets = {
        "reflectivity": 0.999990,
        "transmission": 10.0,
        "absorption": 0.30,
        "thermal_noise": 4.0e-21,
    }
    res = compute_roi_hypervolume(
        designs,
        targets=targets,
        roi_factor=1.5,
        primary_metric="transmission",
    )
    # Target trans 10 * 1.5 = 15 ppm, Abs 0.3 * 1.5 = 0.45, TN 4.0e-21 * 1.5 = 6.0e-21
    # Designs 1 and 2 pass
    assert res["roi_points_count"] == 2
    assert pytest.approx(res["roi_fraction"], 0.01) == 2 / 3


def test_evaluate_dataset_proximity_metrics_transmission():
    designs = create_sample_designs()
    targets = {
        "reflectivity": 0.999990,
        "transmission": 10.0,
        "absorption": 0.30,
        "thermal_noise": 4.0e-21,
        "primary_metric": "transmission",
    }
    metrics = evaluate_dataset_proximity_metrics(
        designs,
        targets=targets,
    )
    assert metrics["yield"]["yield_zero"] > 0
    assert metrics["roi_hypervolume"]["roi_points_count"] == 2
    assert metrics["asf"]["best_index"] == 0
    assert metrics["asf"]["exceeds_all_targets"] is True
