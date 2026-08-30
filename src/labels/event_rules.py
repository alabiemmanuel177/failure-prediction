"""Pure boundary rules for the frozen operational event definitions."""

from __future__ import annotations


def localisation_loss(
    *,
    translation_error_m: float,
    yaw_error_rad: float,
    translation_exceedance_seconds: float,
    yaw_exceedance_seconds: float,
    translation_threshold_m: float = 0.50,
    yaw_threshold_rad: float = 0.50,
    persistence_seconds: float = 2.0,
) -> bool:
    """True only when either channel strictly exceeds its own threshold continuously."""
    translation_lost = (
        translation_error_m > translation_threshold_m
        and translation_exceedance_seconds >= persistence_seconds
    )
    yaw_lost = (
        yaw_error_rad > yaw_threshold_rad
        and yaw_exceedance_seconds >= persistence_seconds
    )
    return translation_lost or yaw_lost


def immobilised(
    *,
    commanded_linear_speed_mps: float,
    planar_progress_m: float,
    continuous_seconds: float,
    command_threshold_mps: float = 0.05,
    maximum_progress_m: float = 0.50,
    persistence_seconds: float = 10.0,
) -> bool:
    """Apply the approved linear-command and planar-displacement interpretation."""
    return (
        abs(commanded_linear_speed_mps) >= command_threshold_mps
        and planar_progress_m < maximum_progress_m
        and continuous_seconds >= persistence_seconds
    )


def unsafe_perception(
    *,
    obstacle_distance_from_footprint_m: float,
    critical_obstacle_missed: bool,
    emergency_intervention: bool,
    protected_region_m: float = 0.42,
) -> bool:
    """Confirm unsafe perception only inside the protected region with intervention."""
    return (
        obstacle_distance_from_footprint_m <= protected_region_m
        and critical_obstacle_missed
        and emergency_intervention
    )
