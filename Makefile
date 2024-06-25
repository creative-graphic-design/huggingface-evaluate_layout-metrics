SRC_DIR := ./github-repo
DST_DIR := ./huggingface-repo

.PHONY: check-vars
check-vars:
ifndef REPO_NAME
	$(error REPO_NAME is not set)
endif

ifndef DIR_NAME
	$(error DIR_NAME is not set)
endif

ifndef HF_USERNAME
	$(error HF_USERNAME is not set)
endif

ifndef HF_EMAIL
	$(error HF_EMAIL is not set)
endif

.PHONY: deploy
deploy: check-vars
	./scripts/push_to_hub.sh
