#!/usr/bin/env python3
"""Create, upload, verify, and optionally publish the core dataset on Zenodo.

The access token is read only from ZENODO_ACCESS_TOKEN. A non-secret local state file
allows an interrupted 142 MB upload to resume without creating duplicate deposits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://zenodo.org/api"
STATE = ROOT / "release/.zenodo-deposition-state.json"
METADATA = ROOT / "release/zenodo_metadata.json"
DEFAULT_TOKEN_FILE = Path.home() / ".config/risk-calibrated-nav/zenodo-token"
FILES = [
    (ROOT / "release/artifacts/research2-failure-prediction-core-v1.tar.zst",
     "research2-failure-prediction-core-v1.tar.zst"),
    (ROOT / "release/CORE_SHA256SUMS", "CORE_SHA256SUMS"),
    (ROOT / "release/DATASET.md", "README.md"),
    (METADATA, "zenodo_metadata.json"),
]


def md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def response_json(response: requests.Response, expected: set[int]) -> dict:
    if response.status_code not in expected:
        try: detail = json.dumps(response.json(), indent=2)
        except ValueError: detail = response.text[:2000]
        raise RuntimeError(f"Zenodo HTTP {response.status_code}: {detail}")
    return response.json()


def save_state(data: dict) -> None:
    STATE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="publish after byte verification")
    ap.add_argument("--dry-run", action="store_true", help="validate local inputs only")
    args = ap.parse_args()

    missing = [str(p) for p, _ in FILES if not p.is_file()]
    if missing: raise SystemExit("missing upload inputs: " + ", ".join(missing))
    expected_sha = (ROOT / "release/CORE_SHA256SUMS").read_text().split()[0]
    archive_sha = sha256(FILES[0][0])
    if archive_sha != expected_sha: raise SystemExit("core archive SHA-256 does not match release/CORE_SHA256SUMS")
    metadata = json.loads(METADATA.read_text())
    if args.dry_run:
        print(json.dumps({"files": {name: path.stat().st_size for path, name in FILES},
                          "archive_sha256": archive_sha, "metadata": metadata}, indent=2))
        return

    token = os.environ.get("ZENODO_ACCESS_TOKEN", "")
    if not token and DEFAULT_TOKEN_FILE.is_file():
        if DEFAULT_TOKEN_FILE.stat().st_mode & 0o077:
            raise SystemExit(f"token file must not be group/world accessible: {DEFAULT_TOKEN_FILE}")
        token = DEFAULT_TOKEN_FILE.read_text().strip()
    if not token:
        raise SystemExit(
            "ZENODO_ACCESS_TOKEN is not set and secure token file is absent: "
            f"{DEFAULT_TOKEN_FILE}"
        )
    headers = {"Authorization": f"Bearer {token}"}
    json_headers = {**headers, "Content-Type": "application/json"}

    if STATE.exists():
        state = json.loads(STATE.read_text()); deposition_id = int(state["deposition_id"])
        deposit = response_json(requests.get(f"{BASE}/deposit/depositions/{deposition_id}", headers=headers, timeout=60), {200})
    else:
        deposit = response_json(requests.post(f"{BASE}/deposit/depositions", headers=json_headers,
                                              data=json.dumps(metadata), timeout=60), {201})
        deposition_id = int(deposit["id"])
        save_state({"deposition_id": deposition_id, "status": "draft-created",
                    "reserved_doi": deposit.get("metadata", {}).get("prereserve_doi", {}).get("doi")})
        print(f"Created Zenodo draft {deposition_id}")

    if deposit.get("submitted"):
        print(json.dumps({"status": "already-published", "doi": deposit.get("doi"),
                          "doi_url": deposit.get("doi_url"), "record_id": deposit.get("record_id")}, indent=2))
        return

    deposit = response_json(requests.put(f"{BASE}/deposit/depositions/{deposition_id}",
                                         headers=json_headers, data=json.dumps(metadata), timeout=60), {200})
    remote = {f.get("filename") or f.get("key"): f for f in deposit.get("files", [])}
    bucket = deposit["links"]["bucket"].rstrip("/")
    for path, name in FILES:
        if name in remote and int(remote[name].get("filesize", remote[name].get("size", -1))) == path.stat().st_size:
            print(f"Already uploaded: {name}"); continue
        print(f"Uploading {name} ({path.stat().st_size / 1_000_000:.1f} MB)")
        with path.open("rb") as stream:
            uploaded = response_json(requests.put(f"{bucket}/{quote(name)}", headers=headers,
                                                   data=stream, timeout=3600), {200, 201})
        remote_md5 = str(uploaded.get("checksum", "")).removeprefix("md5:")
        if int(uploaded.get("size", -1)) != path.stat().st_size or remote_md5 != md5(path):
            raise RuntimeError(f"remote verification failed for {name}")
        print(f"Verified upload: {name}")

    deposit = response_json(requests.get(f"{BASE}/deposit/depositions/{deposition_id}", headers=headers, timeout=60), {200})
    remote = {f.get("filename") or f.get("key"): f for f in deposit.get("files", [])}
    for path, name in FILES:
        if name not in remote or int(remote[name].get("filesize", remote[name].get("size", -1))) != path.stat().st_size:
            raise RuntimeError(f"final Zenodo file inventory mismatch for {name}")
    reserved = deposit.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    save_state({"deposition_id": deposition_id, "status": "verified-draft", "reserved_doi": reserved,
                "files": {name: path.stat().st_size for path, name in FILES}})
    print(f"Verified Zenodo draft {deposition_id}; reserved DOI: {reserved}")

    if not args.publish:
        print("Draft retained. Re-run with --publish to register the DOI.")
        return
    published = response_json(requests.post(f"{BASE}/deposit/depositions/{deposition_id}/actions/publish",
                                            headers=headers, timeout=120), {202})
    result = {"deposition_id": deposition_id, "record_id": published.get("record_id"),
              "doi": published.get("doi"), "doi_url": published.get("doi_url"), "status": "published"}
    save_state(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
