from src.platform_boundary import path_is_locked


def test_exact_and_descendant_paths_are_locked():
    locked = {"src/simulation_worlds", "configs/nav2/nav2_common.yaml"}
    assert path_is_locked("src/simulation_worlds", locked)
    assert path_is_locked("src/simulation_worlds/launch/sim.launch.py", locked)
    assert path_is_locked("configs/nav2/nav2_common.yaml", locked)


def test_operations_and_results_paths_are_not_locked():
    locked = {"src/simulation_worlds", "configs/nav2/nav2_common.yaml"}
    assert not path_is_locked("ops/run_confirmatory_campaign.sh", locked)
    assert not path_is_locked("results/exclusions.log", locked)
    assert not path_is_locked("src/simulation_worlds_extra/file.py", locked)
