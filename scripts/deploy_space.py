"""Build and deploy a metric's Hugging Face Space payload."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

from huggingface_hub import HfApi
from huggingface_hub.repocard import metadata_load


def space_name(metric_dir_name: str) -> str:
    """Return the Space name for a metric directory name."""
    return metric_dir_name.replace("_", "-")


def payload_sources(metric_dir: Path) -> dict[str, Path]:
    """Return the source files and payload names for one metric."""
    metric_name = space_name(metric_dir.name)
    return {
        "README.md": metric_dir / "README.md",
        f"{metric_name}.py": metric_dir / f"{metric_name}.py",
        "app.py": metric_dir / "app.py",
    }


def build_payload(
    metric_dir_name: str,
    source_files: Mapping[str, bytes],
    requirements: str,
) -> dict[str, bytes]:
    """Assemble a validated Space payload without performing I/O."""
    module_name = f"{space_name(metric_dir_name)}.py"
    required_files = ("README.md", module_name, "app.py")
    missing = [name for name in required_files if name not in source_files]
    if missing:
        raise ValueError(f"Missing Space payload files: {', '.join(missing)}")

    empty = [name for name in required_files if not source_files[name].strip()]
    if empty:
        raise ValueError(f"Empty Space payload files: {', '.join(empty)}")
    if not requirements.strip():
        raise ValueError("Exported requirements.txt is empty")

    return {
        **{name: source_files[name] for name in required_files},
        "requirements.txt": requirements.encode(),
    }


def export_requirements(repo_root: Path, package_name: str) -> str:
    """Export locked runtime requirements for a metric package."""
    result = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--package",
            package_name,
            "--no-dev",
            "--no-hashes",
            "--format",
            "requirements-txt",
        ],
        check=True,
        capture_output=True,
        cwd=repo_root,
        text=True,
    )
    return result.stdout


def write_payload(payload_dir: Path, payload: Mapping[str, bytes]) -> None:
    """Write a payload mapping to a directory."""
    payload_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in payload.items():
        (payload_dir / name).write_bytes(contents)


def validate_metadata(payload_dir: Path) -> None:
    """Validate Space metadata without making a network request."""
    metadata = metadata_load(payload_dir / "README.md")
    if not isinstance(metadata, dict):
        raise ValueError("README.md has no metadata block")
    if metadata.get("sdk") != "gradio":
        raise ValueError("README.md must declare sdk: gradio")
    if metadata.get("app_file") != "app.py":
        raise ValueError("README.md must declare app_file: app.py")


def deploy_payload(
    payload_dir: Path,
    namespace: str,
    repo_name: str,
    source_sha: str,
) -> None:
    """Create or update a Gradio Space using the repository token."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN secret is required for deploy")

    repo_id = f"{namespace}/{repo_name}"
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=payload_dir,
        commit_message=f"deploy: {source_sha}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", required=True, help="Metric directory name")
    parser.add_argument("--mode", choices=("validate", "deploy"), required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--source-sha")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if Path(args.metric).name != args.metric:
        raise ValueError("--metric must be a directory name")

    metric_dir = repo_root / "metrics" / args.metric
    if not metric_dir.is_dir():
        raise ValueError(f"Metric directory does not exist: {args.metric}")

    repo_name = space_name(args.metric)
    sources = payload_sources(metric_dir)
    source_files = {name: path.read_bytes() for name, path in sources.items()}
    requirements = export_requirements(repo_root, repo_name)
    payload = build_payload(args.metric, source_files, requirements)

    with tempfile.TemporaryDirectory(prefix="space-payload-") as temp_dir:
        payload_dir = Path(temp_dir)
        write_payload(payload_dir, payload)
        validate_metadata(payload_dir)

        if args.mode == "validate":
            print(f"Validated {args.namespace}/{repo_name}")
            return

        source_sha = args.source_sha
        if source_sha is None:
            source_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                cwd=repo_root,
                text=True,
            ).stdout.strip()
        deploy_payload(payload_dir, args.namespace, repo_name, source_sha)
        print(f"Deployed {args.namespace}/{repo_name} from {source_sha}")


if __name__ == "__main__":
    main()
