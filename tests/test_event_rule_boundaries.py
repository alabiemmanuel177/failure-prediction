from src.labels.event_rules import immobilised, localisation_loss, unsafe_perception


def test_localisation_boundaries_and_interrupted_persistence():
    assert not localisation_loss(
        translation_error_m=0.50, yaw_error_rad=0.0,
        translation_exceedance_seconds=2.0, yaw_exceedance_seconds=0.0,
    )
    assert not localisation_loss(
        translation_error_m=0.51, yaw_error_rad=0.0,
        translation_exceedance_seconds=1.99, yaw_exceedance_seconds=0.0,
    )
    assert localisation_loss(
        translation_error_m=0.51, yaw_error_rad=0.0,
        translation_exceedance_seconds=2.0, yaw_exceedance_seconds=0.0,
    )
    assert localisation_loss(
        translation_error_m=0.0, yaw_error_rad=0.51,
        translation_exceedance_seconds=0.0, yaw_exceedance_seconds=2.0,
    )


def test_immobilisation_boundaries_and_interrupted_persistence():
    assert not immobilised(
        commanded_linear_speed_mps=0.049, planar_progress_m=0.49, continuous_seconds=10.0
    )
    assert not immobilised(
        commanded_linear_speed_mps=0.05, planar_progress_m=0.50, continuous_seconds=10.0
    )
    assert not immobilised(
        commanded_linear_speed_mps=0.05, planar_progress_m=0.49, continuous_seconds=9.99
    )
    assert immobilised(
        commanded_linear_speed_mps=0.05, planar_progress_m=0.49, continuous_seconds=10.0
    )
    assert not immobilised(
        commanded_linear_speed_mps=0.0, planar_progress_m=0.0, continuous_seconds=20.0
    )


def test_unsafe_perception_protected_region_boundary():
    assert not unsafe_perception(
        obstacle_distance_from_footprint_m=0.421,
        critical_obstacle_missed=True,
        emergency_intervention=True,
    )
    assert unsafe_perception(
        obstacle_distance_from_footprint_m=0.42,
        critical_obstacle_missed=True,
        emergency_intervention=True,
    )
    assert not unsafe_perception(
        obstacle_distance_from_footprint_m=0.41,
        critical_obstacle_missed=True,
        emergency_intervention=False,
    )
