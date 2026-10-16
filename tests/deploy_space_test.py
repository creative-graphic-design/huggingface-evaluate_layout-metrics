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
