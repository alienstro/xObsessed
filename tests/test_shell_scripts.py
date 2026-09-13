import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


@pytest.mark.parametrize("name", ["pod.sh", "watchdog.sh", "quantize.sh"])
def test_the_script_is_valid_bash(name):
    subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


def test_the_watchdog_stops_on_error_and_never_deletes_the_pod():
    text = (SCRIPTS / "watchdog.sh").read_text()
    assert "set -euo pipefail" in text
    assert "scripts/runpod.py stop" in text
    assert "terminate --yes" not in text


def test_quantization_accepts_a_name_and_limits_build_jobs():
    text = (SCRIPTS / "quantize.sh").read_text()
    assert 'NAME="${3:-' in text
    assert "patch_llama_cpp.py" not in text
    assert 'BUILD_JOBS="${BUILD_JOBS:-1}"' in text


def test_the_pod_helper_reports_a_missing_connection(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "POD_SSH"}
    env["ENV_FILE"] = str(tmp_path / "absent.env")
    result = subprocess.run(
        ["bash", str(SCRIPTS / "pod.sh"), "check"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "POD_SSH is empty" in result.stderr
    assert "unbound variable" not in result.stderr


def test_the_pod_helper_has_a_separate_data_transfer():
    text = (SCRIPTS / "pod.sh").read_text()
    assert "data)" in text
    assert "/workspace/xFrieren" in text
    assert "--delete" not in text


def test_the_gitignore_holds_the_secret_file():
    assert ".env" in (ROOT / ".gitignore").read_text()


def test_the_card_states_the_limits_without_false_success_claims():
    from xfrieren.card import build_card

    card = build_card("xfrieren-1.7b", "user/repo")
    for value in ("invents", "fan work", "non-commercial", "--jinja", "system_prompt.txt"):
        assert value in card.lower()
    assert "unverified" in card.lower()


def test_remote_arguments_remain_literal(tmp_path):
    ssh = tmp_path / "ssh"
    log = tmp_path / "ssh.json"
    ssh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['SSH_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    ssh.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               ENV_FILE=str(tmp_path / "absent.env"), POD_SSH="root@192.0.2.1",
               SSH_LOG=str(log))
    argument = "a'b $(touch /tmp/xfrieren-shell-must-not-run)"
    subprocess.run(
        ["bash", str(SCRIPTS / "pod.sh"), "run", "printf", "%s", argument],
        env=env, check=True,
    )
    remote = json.loads(log.read_text())[-1]
    # Parse each shell layer without an SSH connection or a tmux process.
    script = "tmux() { printf '%s\\0' \"$@\"; }; " + remote.replace(
        "cd '/workspace/xFrieren' && mkdir -p out && ", ""
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, check=True)
    command = result.stdout.decode().split("\0")[-2]
    assert "bash" in command
    assert not Path("/tmp/xfrieren-shell-must-not-run").exists()
    import shlex

    layers = shlex.split(command)
    assert layers[:3] == ["bash", "-o", "pipefail"]
    inner = shlex.split(layers[4])
    assert argument in inner
