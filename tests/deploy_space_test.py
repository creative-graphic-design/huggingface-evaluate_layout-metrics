import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "deploy_space.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("deploy_space", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
assert SCRIPT_SPEC.loader is not None
deploy_space = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(deploy_space)

build_payload = deploy_space.build_payload
deploy_payload = deploy_space.deploy_payload
space_name = deploy_space.space_name


class StubHfApi:
    def __init__(self, exists: bool, create_error: Exception | None = None) -> None:
        self.exists = exists
        self.create_error = create_error
        self.calls: list[tuple[str, object]] = []

    def repo_exists(self, repo_id: str, *, repo_type: str) -> bool:
        self.calls.append(("repo_exists", {"repo_id": repo_id, "repo_type": repo_type}))
        return self.exists

    def create_repo(self, **kwargs: object) -> None:
        self.calls.append(("create_repo", kwargs))
        if self.create_error is not None:
            raise self.create_error

    def upload_folder(self, **kwargs: object) -> None:
        self.calls.append(("upload_folder", kwargs))


def test_build_payload_has_space_files_requirements_and_mapped_name() -> None:
    metric_dir_name = "layout_example_metric"
    sources = {
        "README.md": b"---\nsdk: gradio\n---\nREADME",
        "layout-example-metric.py": b"METRIC = True\n",
        "app.py": b"demo = True\n",
    }
    payload = build_payload(
        metric_dir_name,
        sources,
        "evaluate==0.4.7.dev0\n",
    )

    assert space_name(metric_dir_name) == "layout-example-metric"
    assert payload == {
        "README.md": sources["README.md"],
        "layout-example-metric.py": sources["layout-example-metric.py"],
        "app.py": sources["app.py"],
        "requirements.txt": b"evaluate==0.4.7.dev0\n",
    }


@pytest.mark.parametrize(
    ("sources", "requirements", "message"),
    [
        (
            {"README.md": b"README", "app.py": b"app"},
            "evaluate==0.4.7.dev0\n",
            "Missing Space payload files: layout-example-metric.py",
        ),
        (
            {
                "README.md": b"",
                "layout-example-metric.py": b"module",
                "app.py": b"app",
            },
            "evaluate==0.4.7.dev0\n",
            "Empty Space payload files: README.md",
        ),
        (
            {
                "README.md": b"README",
                "layout-example-metric.py": b"module",
                "app.py": b"app",
            },
            "\n",
            "Exported requirements.txt is empty",
        ),
    ],
)
def test_build_payload_rejects_invalid_inputs(
    sources: dict[str, bytes], requirements: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_payload("layout_example_metric", sources, requirements)


def test_deploy_payload_requires_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(SystemExit, match="HF_TOKEN secret is required for deploy"):
        deploy_payload(tmp_path, "creative-graphic-design", "layout-example", "sha")


def test_deploy_existing_space_skips_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = StubHfApi(exists=True)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(deploy_space, "HfApi", lambda token: api)

    deploy_payload(tmp_path, "creative-graphic-design", "layout-example", "sha")

    assert [name for name, _ in api.calls] == ["repo_exists", "upload_folder"]


def test_deploy_missing_space_creates_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = StubHfApi(exists=False)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(deploy_space, "HfApi", lambda token: api)

    deploy_payload(tmp_path, "creative-graphic-design", "layout-example", "sha")

    assert [name for name, _ in api.calls] == [
        "repo_exists",
        "create_repo",
        "upload_folder",
    ]
    create_kwargs = api.calls[1][1]
    assert create_kwargs == {
        "repo_id": "creative-graphic-design/layout-example",
        "repo_type": "space",
        "space_sdk": "gradio",
        "exist_ok": True,
    }


def test_deploy_creation_failure_exits_with_owner_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = StubHfApi(
        exists=False,
        create_error=deploy_space.HfHubHTTPError(
            "402 Client Error: Payment Required",
            server_message="Space hosting plan required",
        ),
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(deploy_space, "HfApi", lambda token: api)

    with pytest.raises(SystemExit) as error:
        deploy_payload(tmp_path, "creative-graphic-design", "layout-example", "sha")

    assert "Space hosting plan required" in str(error.value)
    assert "must be created by the owner (plan or manual)" in str(error.value)
    assert [name for name, _ in api.calls] == ["repo_exists", "create_repo"]
