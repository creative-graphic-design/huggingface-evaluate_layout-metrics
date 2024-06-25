#!/usr/bin/env bash

set -eux

function main() {
    script_name=${REPO_NAME}.py

    ls ${SRC_DIR}/${DIR_NAME}
    
    # Copy the script to the destination directory
    cp ${SRC_DIR}/${DIR_NAME}/README.md ${DST_DIR}/README.md
    cp ${SRC_DIR}/${DIR_NAME}/${script_name} ${DST_DIR}/${script_name}
    
    # Set configuration for git
    git -C ${DST_DIR} config user.name ${HF_USERNAME}
    git -C ${DST_DIR} config user.email ${HF_EMAIL}
    
    # Add files
    git -C ${DST_DIR} add README.md requirements.txt ${script_name}
    
    if git -C ${DST_DIR} diff --cached --quiet; then
        echo "No changes to commit"
        exit 0
    fi
    
    # Commit using the latest commit hash of the source repository
    msg=$(git -C ${SRC_DIR} rev-parse HEAD)
    git -C ${DST_DIR} commit -m "deploy: ${msg}"
    
    # Push to Huggingface Space only if `CI` is `true` and `GITHUB_EVENT_NAME` is not `pull_request`
    if [[ "${CI}" == "true" ]] && [[ "${GITHUB_EVENT_NAME}" != "pull_request" ]]; then
        git -C ${DST_DIR} push -u origin main
    else
        git -C ${DST_DIR} push --dry-run -u origin main
    fi
}

main
