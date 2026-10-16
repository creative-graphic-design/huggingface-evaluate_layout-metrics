from pathlib import Path

from scripts.deploy_space import build_payload, payload_sources, space_name


def test_build_payload_has_space_files_requirements_and_mapped_name(
    tmp_path: Path,
) -> None:
    metric_dir = tmp_path / "layout_example_metric"
    metric_dir.mkdir()
    for path in payload_sources(metric_dir).values():
        path.write_text("payload", encoding="utf-8")

    sources = {
        name: path.read_bytes() for name, path in payload_sources(metric_dir).items()
    }
    payload = build_payload(
        metric_dir.name,
        sources,
        "evaluate==0.4.7.dev0\n",
    )

    assert space_name(metric_dir.name) == "layout-example-metric"
    assert set(payload) == {
        "README.md",
        "layout-example-metric.py",
        "app.py",
        "requirements.txt",
    }
    assert payload["requirements.txt"].strip()
