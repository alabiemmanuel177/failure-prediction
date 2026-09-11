"""Validate the immutable Research 1 scientific-platform boundary."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any


class BoundaryError(RuntimeError):
    """The checked-out Research 1 platform does not match the frozen boundary."""


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def path_is_locked(path: str, locked_paths: set[str]) -> bool:
    normalized = path.strip("/")
    return any(normalized == item or normalized.startswith(item.rstrip("/") + "/")
               for item in locked_paths)


def _is_tree(repo: Path, object_id: str) -> bool:
    try:
        return git_output(repo, "cat-file", "-t", object_id) == "tree"
    except subprocess.CalledProcessError:
        return False


def _tree_blobs(repo: Path, tree: str) -> dict[str, str]:
    blobs: dict[str, str] = {}
    for line in git_output(repo, "ls-tree", "-r", tree).splitlines():
        meta, _, path = line.partition("\t")
        blobs[path] = meta.split()[2]
    return blobs


def tree_blob_changes(repo: Path, expected_tree: str, actual_tree: str) -> list[str]:
    """Pinned blobs of ``expected_tree`` that are modified or missing in ``actual_tree``.

    Additions in ``actual_tree`` are not reported: they cannot alter any content the
    frozen platform serves, whereas a modified or deleted pinned blob can.
    """
    expected = _tree_blobs(repo, expected_tree)
    actual = _tree_blobs(repo, actual_tree)
    return sorted(
        f"{path} ({'deleted' if path not in actual else 'modified'})"
        for path, blob in expected.items() if actual.get(path) != blob
    )


def dirty_paths(repo: Path) -> set[str]:
    commands = (
        ("diff", "--name-only", "HEAD"),
        ("diff", "--cached", "--name-only", "HEAD"),
        ("ls-files", "--others", "--exclude-standard"),
    )
    paths: set[str] = set()
    for command in commands:
        output = git_output(repo, *command)
        paths.update(line for line in output.splitlines() if line)
    return paths


def validate_platform(lock: dict[str, Any]) -> tuple[Path, str]:
    research1 = Path(lock["research1_root"]).resolve()
    boundary_commit = str(lock["commit_sha"])
    actual_commit = git_output(research1, "rev-parse", "HEAD")
    if lock.get("boundary_mode") != "git_path_objects":
        if actual_commit != boundary_commit:
            raise BoundaryError(
                f"Research 1 commit changed: locked {boundary_commit}, actual {actual_commit}"
            )
        return research1, actual_commit

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", boundary_commit, actual_commit],
        cwd=research1, check=False, capture_output=True, text=True,
    )
    if ancestor.returncode != 0:
        raise BoundaryError(
            f"Research 1 HEAD {actual_commit} does not descend from G6 boundary "
            f"{boundary_commit}"
        )

    locked = {str(path): str(object_id)
              for path, object_id in lock.get("locked_git_objects", {}).items()}
    if not locked:
        raise BoundaryError("content boundary has no locked_git_objects")
    mismatches: list[str] = []
    for path, expected in sorted(locked.items()):
        try:
            actual = git_output(research1, "rev-parse", f"HEAD:{path}")
        except subprocess.CalledProcessError:
            actual = "missing"
        if actual == expected:
            continue
        # A pinned directory tree whose hash changed only because files were ADDED
        # (Research 1 follow-up work, 8 September 2026) keeps every blob Research 2
        # depends on byte-identical; a modified or deleted pinned blob still refuses.
        if actual != "missing" and _is_tree(research1, expected):
            changed = tree_blob_changes(research1, expected, actual)
            if not changed:
                continue
            mismatches.append(f"{path}: pinned content changed or removed: " + ", ".join(changed[:10]))
            continue
        mismatches.append(f"{path}: expected {expected}, actual {actual}")
    if mismatches:
        raise BoundaryError("Research 1 shared-platform content changed:\n- " +
                            "\n- ".join(mismatches))

    modified_locked = sorted(
        path for path in dirty_paths(research1) if path_is_locked(path, set(locked))
    )
    if modified_locked:
        raise BoundaryError(
            "Research 1 has uncommitted changes in locked platform paths: "
            + ", ".join(modified_locked)
        )
    return research1, actual_commit
