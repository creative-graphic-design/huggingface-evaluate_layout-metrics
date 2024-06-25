#!/usr/bin/env bash

set -eux

function main() {
    script_name=${REPO_NAME}.py

	cp ${SRC_DIR}/${DIR_NAME}/README.md ${DST_DIR}/README.md
	cp ${SRC_DIR}/${DIR_NAME}/${script_name} ${DST_DIR}/${script_name}

	git -C ${DST_DIR} config user.name $(HF_USERNAME)
	git -C ${DST_DIR} config user.email $(HF_EMAIL)

	git -C ${DST_DIR} add README.md requirements.txt ${script_name}

	if git -C ${DST_DIR} diff --cached --quiet; then
		echo "No changes to commit"
	else
		msg=$(git -C ${SRC_DIR} rev-parse HEAD)
		git -C ${DST_DIR} commit -m "deploy: ${msg}"
		git -C $(DST_DIR) push --dry-run -u origin main
	fi
}

main
