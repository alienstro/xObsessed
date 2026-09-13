"""Read or stop a RunPod pod. Deletion requires confirmation and upload verification."""

import argparse
import json
import os
import urllib.error
import urllib.request

BASE_URL = "https://rest.runpod.io/v1"


def build_request(method, path, token):
    """Return a request for the RunPod REST API."""
    return urllib.request.Request(
        f"{BASE_URL}/{path}",
        method=method,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "xfrieren/0.1"},
    )


def send(request):
    """Send one request with a timeout."""
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as error:
        raise SystemExit(f"RunPod returned HTTP {error.code}.") from error


def guard_terminate(pod_id, confirmed):
    """Require a pod ID and explicit consent before deletion."""
    if not pod_id:
        raise SystemExit("RUNPOD_POD_ID is empty.")
    if not confirmed:
        raise SystemExit("Warning: termination deletes the pod disk. Pass --yes only after verification.")
    return pod_id


def ssh_string_for(pod, key_path="~/.ssh/id_ed25519"):
    """Return the current direct SSH connection."""
    port = (pod.get("portMappings") or {}).get("22")
    if not port or not pod.get("publicIp"):
        raise SystemExit("The pod needs a public IP and a direct TCP port for SSH.")
    return f"root@{pod['publicIp']} -p {port} -i {key_path}"


def main():
    parser = argparse.ArgumentParser(description="Read, stop, or explicitly delete a RunPod pod.")
    parser.add_argument("command", choices=["list", "ssh", "stop", "terminate"])
    parser.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID"))
    parser.add_argument("--yes", action="store_true")
    arguments = parser.parse_args()
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        parser.error("set RUNPOD_API_KEY")
    if arguments.command == "list":
        payload = send(build_request("GET", "pods", token))
        pods = payload if isinstance(payload, list) else payload.get("data", [])
        for pod in pods:
            print(f"{pod.get('id')} {pod.get('name')} {pod.get('desiredStatus')}")
        return
    if not arguments.pod_id:
        parser.error("set RUNPOD_POD_ID or pass --pod-id")
    path = f"pods/{arguments.pod_id}"
    if arguments.command == "ssh":
        print(ssh_string_for(send(build_request("GET", path, token))))
    elif arguments.command == "stop":
        send(build_request("POST", f"{path}/stop", token))
        print(f"Requested a stop for {arguments.pod_id}. Check storage charges in RunPod.")
    else:
        guard_terminate(arguments.pod_id, arguments.yes)
        from verify_upload import MANIFEST_PATH, verify_manifest
        from huggingface_hub import HfApi

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        faults = verify_manifest(HfApi(token=os.environ["HF_TOKEN"]), os.environ["HF_REPO_ID"], manifest)
        if faults:
            raise SystemExit(f"Cannot delete the pod: upload verification failed: {faults}")
        send(build_request("DELETE", path, token))
        print(f"Requested termination for {arguments.pod_id}.")


if __name__ == "__main__":
    main()
