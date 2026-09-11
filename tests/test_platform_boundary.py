import pytest
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


def test_pinned_tree_accepts_additions_but_refuses_modified_or_deleted_blobs(tmp_path):
    """Research 1 follow-up work may add files under a pinned directory; every blob the
    boundary pinned must stay byte-identical and present."""
    import subprocess
    from src.platform_boundary import tree_blob_changes, validate_platform, BoundaryError
    repo = tmp_path / "r1"
    repo.mkdir()
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (repo / "configs/routes").mkdir(parents=True)
    (repo / "configs/routes/test_00.yaml").write_text("a: 1\n")
    git("add", "."); git("commit", "-qm", "boundary")
    boundary = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD:configs/routes")
    lock = {"research1_root": str(repo), "commit_sha": boundary, "boundary_mode": "git_path_objects",
            "locked_git_objects": {"configs/routes": tree}}
    # addition only: accepted
    (repo / "configs/routes/dev_60000.yaml").write_text("b: 2\n")
    git("add", "."); git("commit", "-qm", "add follow-up route")
    assert tree_blob_changes(repo, tree, git("rev-parse", "HEAD:configs/routes")) == []
    validate_platform(lock)
    # modification of a pinned blob: refused
    (repo / "configs/routes/test_00.yaml").write_text("a: 2\n")
    git("add", "."); git("commit", "-qm", "modify pinned route")
    with pytest.raises(BoundaryError, match="test_00.yaml \\(modified\\)"):
        validate_platform(lock)
    # deletion of a pinned blob: refused
    git("rm", "-q", "configs/routes/test_00.yaml"); git("commit", "-qm", "delete pinned route")
    with pytest.raises(BoundaryError, match="deleted"):
        validate_platform(lock)
