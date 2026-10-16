#!/usr/bin/env bash

set -euo pipefail

: "${DIR_NAME:?DIR_NAME is required}"
: "${SPACE_NAMESPACE:?SPACE_NAMESPACE is required}"
MODE="${MODE:-validate}"
SRC_DIR="${SRC_DIR:-.}"
REPO_NAME="${DIR_NAME//_/-}"
PAYLOAD_DIR="$(mktemp -d "${RUNNER_TEMP:-/tmp}/space-payload.XXXXXX")"
trap 'rm -rf "${PAYLOAD_DIR}"' EXIT

cp "${SRC_DIR}/metrics/${DIR_NAME}/README.md" "${PAYLOAD_DIR}/README.md"
cp "${SRC_DIR}/metrics/${DIR_NAME}/${REPO_NAME}.py" "${PAYLOAD_DIR}/${REPO_NAME}.py"
cp "${SRC_DIR}/metrics/${DIR_NAME}/app.py" "${PAYLOAD_DIR}/app.py"
uv export --frozen --package "${REPO_NAME}" --no-dev --no-hashes --format requirements-txt > "${PAYLOAD_DIR}/requirements.txt"

test -s "${PAYLOAD_DIR}/README.md"
test -s "${PAYLOAD_DIR}/${REPO_NAME}.py"
test -s "${PAYLOAD_DIR}/app.py"
test -s "${PAYLOAD_DIR}/requirements.txt"

uv run --frozen --no-dev python - "${PAYLOAD_DIR}" "${MODE}" "${SPACE_NAMESPACE}" "${REPO_NAME}" "${SOURCE_SHA:-$(git -C "${SRC_DIR}" rev-parse HEAD)}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.repocard import metadata_load

payload_dir = Path(sys.argv[1])
mode = sys.argv[2]
namespace = sys.argv[3]
repo_name = sys.argv[4]
source_sha = sys.argv[5]
metadata = metadata_load(payload_dir / "README.md")

if not isinstance(metadata, dict):
    raise SystemExit("README.md has no metadata block")
if metadata.get("sdk") != "gradio":
    raise SystemExit("README.md must declare sdk: gradio")
if metadata.get("app_file") != "app.py":
    raise SystemExit("README.md must declare app_file: app.py")

if mode == "validate":
    print(f"Validated {namespace}/{repo_name}")
    raise SystemExit(0)
if mode != "deploy":
    raise SystemExit(f"Unknown MODE: {mode}")

token = os.environ["HF_TOKEN"]
repo_id = f"{namespace}/{repo_name}"
api = HfApi(token=token)
api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
    token=token,
)
api.upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=payload_dir,
    commit_message=f"deploy: {source_sha}",
    token=token,
)
print(f"Deployed {repo_id} from {source_sha}")
PY
