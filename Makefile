SRC_DIR := ./github-repo
DST_DIR := ./huggingface-repo

.PHONY: check-vars
check-vars:
ifndef ($(REPO_NAME),)
	$(error REPO_NAME is not set)
endif

ifndef ($(DIR_NAME),)
	$(error DIR_NAME is not set)
endif

ifndef ($(HF_USERNAME),)
	$(error HF_USERNAME is not set)
endif

ifndef ($(HF_EMAIL),)
	$(error HF_EMAIL is not set)
endif

.PHONY: deploy
deploy: check-vars
	script_name=$(REPO_NAME).py

	cp $(SRC_DIR)/$(DIR_NAME)/README.md $(DST_DIR)/README.md | true
	cp $(SRC_DIR)/$(DIR_NAME)/${script_name} $(DST_DIR)/${script_name}

	git -C $(DST_DIR) config user.name $(HF_USERNAME)
	git -C $(DST_DIR) config user.email $(HF_EMAIL)

	git -C $(DST_DIR) add README.md requirements.txt ${script_name}

	if git -C $(DST_DIR) diff --cached --quiet; then
		echo "No changes to commit"
	else
		msg=$(git -C $(SRC_DIR) rev-parse HEAD)
		git -C $(DST_DIR) commit -m "deploy: ${msg}"
		git -C $(DST_DIR) push -u origin main
	fi
