import hashlib
import json
from types import SimpleNamespace

import pytest

from push_to_hub import build_manifest
from runpod import guard_terminate, ssh_string_for
from verify_upload import REQUIRED, check_files, verify_manifest


def test_the_required_files_exclude_xslm_artifacts():
    expected = [name.format(name="xfrieren-1.7b") for name in REQUIRED]
    assert check_files(expected, "xfrieren-1.7b") == []
    assert "tokenizer_config.json" in expected
    assert not any("instruct/" in name for name in expected)


def test_the_verifier_reports_each_missing_file():
    assert "model.safetensors" in check_files(["config.json"], "xfrieren-1.7b")


def test_a_manifest_hashes_current_files(tmp_path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"new weights")
    manifest = build_manifest({"model.safetensors": source}, "xfrieren-1.7b", "run-1")
    assert manifest["run_id"] == "run-1"
    assert manifest["files"]["model.safetensors"] == {
        "sha256": hashlib.sha256(b"new weights").hexdigest(),
        "size": 11,
    }


def test_the_verifier_rejects_a_stale_run_before_files():
    manifest = {"run_id": "new", "model_name": "xfrieren-1.7b", "files": {}}
    api = SimpleNamespace(repo_info=lambda **kwargs: SimpleNamespace(sha="revision"))
    remote = json.dumps({**manifest, "run_id": "old"}).encode()
    faults = verify_manifest(api, "user/repo", manifest, read_remote=lambda *args: remote)
    assert "manifest_mismatch" in faults


def test_the_verifier_rejects_a_wrong_weight_hash():
    files = {
        name.format(name="xfrieren-1.7b"): {"size": 1, "sha256": "expected"}
        for name in REQUIRED
    }
    manifest = {"run_id": "run-1", "model_name": "xfrieren-1.7b", "files": files}
    api = SimpleNamespace(
        repo_info=lambda **kwargs: SimpleNamespace(sha="revision"),
        get_paths_info=lambda **kwargs: [
            SimpleNamespace(path=path, size=1, lfs=SimpleNamespace(sha256="wrong"))
            for path in kwargs["paths"]
        ],
    )
    faults = verify_manifest(
        api, "user/repo", manifest,
        read_remote=lambda *args: json.dumps(manifest).encode(),
    )
    assert "hash_mismatch:model.safetensors" in faults


def test_the_verifier_accepts_current_hashes():
    digest = hashlib.sha256(b"x").hexdigest()
    files = {name.format(name="xfrieren-1.7b"): {"size": 1, "sha256": digest} for name in REQUIRED}
    manifest = {"run_id": "run-1", "model_name": "xfrieren-1.7b", "files": files}
    api = SimpleNamespace(
        repo_info=lambda **kwargs: SimpleNamespace(sha="revision"),
        get_paths_info=lambda **kwargs: [
            SimpleNamespace(path=path, size=1, lfs=SimpleNamespace(sha256=digest))
            for path in kwargs["paths"]
        ],
    )
    assert verify_manifest(
        api, "user/repo", manifest, read_remote=lambda *args: json.dumps(manifest).encode()
    ) == []


@pytest.mark.parametrize("pod_id,confirmed", [("", True), ("pod-1", False)])
def test_termination_needs_a_pod_and_confirmation(pod_id, confirmed):
    with pytest.raises(SystemExit):
        guard_terminate(pod_id, confirmed)


def test_the_ssh_string_uses_the_current_public_port():
    assert ssh_string_for({"publicIp": "192.0.2.1", "portMappings": {"22": 10022}}) == (
        "root@192.0.2.1 -p 10022 -i ~/.ssh/id_ed25519"
    )
