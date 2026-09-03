from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

import aoi.tools.build_submission as submission


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_bundle(root: Path, run: Path, category: str, marker: bytes) -> None:
    artifacts = {
        "model.pt": b"model-" + marker,
        "calibration.yaml": b"threshold: 0.5\n",
        "support_receipt.yaml": b"normal: 100\ndefect: 30\n",
    }
    for name, value in artifacts.items():
        (run / name).write_bytes(value)
    source = (root / "assets" / "source_model.pt").read_bytes()
    runtime_asset = (root / "assets" / "runtime_asset.bin").read_bytes()
    bundle = {
        "schema_version": 3,
        "target_dataset": "custom",
        "target_object": category,
        "source_model": "assets/source_model.pt",
        "models": {"primary": "model.pt"},
        "calibration": "calibration.yaml",
        "support_receipt": "support_receipt.yaml",
        "assets": {
            "resnet18_weights": {
                "path": "assets/runtime_asset.bin",
                "sha256": _digest(runtime_asset),
            },
            "dinov2_weights": {
                "path": "assets/runtime_asset.bin",
                "sha256": _digest(runtime_asset),
            },
            "dinov2_repository": {
                "path": "assets/runtime_repo",
                "sha256": None,
            },
            "source_model": {
                "path": "assets/source_model.pt",
                "sha256": _digest(source),
            },
        },
        "artifact_sha256": {
            "source_model": _digest(source),
            "models": {"primary": _digest(artifacts["model.pt"])},
            "calibration": _digest(artifacts["calibration.yaml"]),
            "support_receipt": _digest(artifacts["support_receipt.yaml"]),
        },
        "bundle_hash": "pending",
    }
    bundle["bundle_hash"] = submission._payload_hash(
        {key: value for key, value in bundle.items() if key != "bundle_hash"}
    )
    (run / "model_bundle.yaml").write_text(
        yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8"
    )


def _project(tmp_path: Path) -> Path:
    for directory in (
        "aoi/cli",
        "aoi/feedback",
        "aoi/runtime",
        "tests",
        "configs",
        "assets",
        "assets/runtime_repo",
        "data/unrelated_demo",
        "outputs/task/run_a",
        "outputs/task/run_b",
        "outputs/evidence",
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "train.py").write_text("print('train')\n", encoding="utf-8")
    (tmp_path / "aoi" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "aoi" / "feedback" / "__init__.py").write_text(
        '''class FeedbackStore:
    def __init__(self, root):
        self.root = root


class _State:
    def receipt(self):
        return {"enabled": True, "revision_id": None}


def apply_active_feedback(runtime, store, category, *, enabled=True):
    assert enabled is True
    return _State()
''',
        encoding="utf-8",
    )
    (tmp_path / "aoi" / "paths.py").write_text(
        """from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_artifact(value, *, base=None, strict=True):
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else []
    if base is not None and not raw.is_absolute():
        candidates.append(Path(base) / raw)
    if not raw.is_absolute():
        candidates.append(PROJECT_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if strict:
        raise FileNotFoundError(value)
    return candidates[0].resolve()
""",
        encoding="utf-8",
    )
    (tmp_path / "aoi" / "runtime" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "aoi" / "cli" / "__init__.py").write_text("", encoding="utf-8")
    configured_runner = """from pathlib import Path
import yaml

def run(config):
    payload = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    marker = Path(payload["marker"])
    marker.write_text(MODE, encoding="utf-8")
    print(MODE)
    return 0
"""
    (tmp_path / "aoi" / "cli" / "custom_training.py").write_text(
        "MODE = 'adapted'\n" + configured_runner,
        encoding="utf-8",
    )
    (tmp_path / "aoi" / "cli" / "custom_evaluation.py").write_text(
        "MODE = 'evaluated'\n" + configured_runner,
        encoding="utf-8",
    )
    (tmp_path / "aoi" / "runtime" / "config.py").write_text(
        "def resolve_device(value):\n    return str(value)\n", encoding="utf-8"
    )
    (tmp_path / "aoi" / "runtime" / "data.py").write_text(
        "from pathlib import Path\n\ndef read_rgb(path):\n    return Path(path).read_bytes()\n",
        encoding="utf-8",
    )
    (tmp_path / "aoi" / "runtime" / "runtime.py").write_text(
        """class AdaptedRuntime:
    def __init__(self, bundle, *, device):
        self.bundle = str(bundle)
        self.device = device

    def predict(self, image, sample_id):
        return {"prediction": False, "bytes_read": len(image), "sample_id": sample_id,
                "device": self.device}
""",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "test.yaml").write_text(
        "minimum_total_images: 8050\n", encoding="utf-8"
    )
    (tmp_path / "assets" / "source_model.pt").write_bytes(b"source")
    (tmp_path / "assets" / "runtime_asset.bin").write_bytes(b"runtime")
    (tmp_path / "assets" / "unrelated.bin").write_bytes(b"unrelated")
    (tmp_path / "assets" / "runtime_repo" / "loader.py").write_text(
        "RUNTIME = True\n", encoding="utf-8"
    )
    (tmp_path / "data" / "unrelated_demo" / "demo.png").write_bytes(b"demo")
    _write_bundle(
        tmp_path,
        tmp_path / "outputs" / "task" / "run_a",
        "part_a",
        b"a",
    )
    _write_bundle(
        tmp_path,
        tmp_path / "outputs" / "task" / "run_b",
        "part_b",
        b"b",
    )
    for run_name in ("run_a", "run_b"):
        (tmp_path / "outputs" / "task" / run_name / "unreferenced.txt").write_text(
            "must not be packaged\n", encoding="utf-8"
        )
    pointer = {
        "schema_version": 1,
        "dataset": "custom",
        "protocol": "private-100+30",
        "models": {
            "part_a": "run_a/model_bundle.yaml",
            "part_b": "run_b/model_bundle.yaml",
        },
        "planned_independent_test_images": 8050,
        "skipped_categories": [],
    }
    (tmp_path / "outputs" / "task" / "active_models.yaml").write_text(
        yaml.safe_dump(pointer, sort_keys=False), encoding="utf-8"
    )
    return tmp_path / "outputs" / "task" / "active_models.yaml"


def test_submission_contains_only_selected_model_and_package_local_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "submission.zip"

    result = submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        maximum_bytes=1_000_000,
    )

    assert result["under_size_limit"] is True
    assert result["verification"]["package_sha256_verified"] is True
    assert result["verification"]["bundle_hash_verified"] is True
    assert result["verification"]["artifact_hashes_verified"] is True
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "outputs/task/run_a/model.pt" in names
        assert "assets/source_model.pt" in names
        assert "assets/runtime_asset.bin" in names
        assert "assets/runtime_repo/loader.py" in names
        assert not any("run_b" in name for name in names)
        assert not any(name.startswith("data/") for name in names)
        assert "assets/unrelated.bin" not in names
        assert "outputs/task/run_a/unreferenced.txt" not in names
        assert "configs/submission_task.yaml" in names
        assert "configs/train.yaml" in names
        assert "configs/custom_training_base.yaml" in names
        assert "configs/custom_train.example.yaml" in names
        assert "configs/custom_test.example.yaml" in names
        assert "configs/feedback.yaml" in names
        assert "configs/benchmark_official_rtx2060.yaml" in names
        assert "configs/benchmark_internal_rtx4070.yaml" in names
        assert "configs/benchmark_cpu_diagnostic.yaml" in names
        assert submission.PACKAGE_MODEL_GUIDE_NAME in names
        assert submission.PACKAGE_NOTICES_NAME in names
        assert submission.PACKAGE_UI_LAUNCHER_NAME in names
        assert b".venv\\Scripts\\python.exe" in archive.read(
            submission.PACKAGE_UI_LAUNCHER_NAME
        )
        assert submission.PACKAGE_RUNNER_NAME in names
        assert "configs/test.yaml" not in names
        assert submission.MANIFEST_NAME in names

        filtered = yaml.safe_load(
            archive.read("outputs/task/active_models.yaml").decode("utf-8")
        )
        assert set(filtered["models"]) == {"part_a"}
        assert filtered["models"]["part_a"] == "run_a/model_bundle.yaml"
        assert "planned_independent_test_images" not in filtered
        assert (
            filtered["submission_selection"]["pooled_pointer_metadata_removed"] is True
        )

        config = yaml.safe_load(
            archive.read("configs/submission_task.yaml").decode("utf-8")
        )
        assert config["category"] == "part_a"
        assert config["active_models_pointer"] == "outputs/task/active_models.yaml"
        assert config["model_bundle"] == "outputs/task/run_a/model_bundle.yaml"
        assert config["execution"]["training_supported"] is True
        assert config["execution"]["evaluation_supported"] is True
        assert config["execution"]["feedback_supported"] is True
        assert config["execution"]["active_learning_query_supported"] is True
        assert config["execution"]["interactive_inference_feedback_enabled"] is True
        assert config["execution"]["interactive_inference_feedback_default"] is True
        assert config["execution"]["interactive_inference_no_feedback_supported"] is True
        assert config["execution"]["private_evaluation_feedback_frozen"] is True

        feedback_config = yaml.safe_load(archive.read("configs/feedback.yaml"))
        assert feedback_config["active_models_pointer"] == "outputs/task/active_models.yaml"
        assert feedback_config["category"] == "part_a"
        assert feedback_config["query_budget"] == 20
        assert feedback_config["active_learning"]["weights"] == {
            "uncertainty": 0.55,
            "branch_disagreement": 0.20,
            "rarity": 0.25,
        }

        train_config = yaml.safe_load(archive.read("configs/train.yaml"))
        assert archive.read("configs/custom_training_base.yaml") == archive.read(
            "configs/train.yaml"
        )
        assert train_config["source_assets"] == {
            "dinov2_repository": "assets/runtime_repo",
            "dinov2_weights": "assets/runtime_asset.bin",
            "resnet18_weights": "assets/runtime_asset.bin",
            "source_model": "assets/source_model.pt",
        }
        custom_test = yaml.safe_load(archive.read("configs/custom_test.example.yaml"))
        assert custom_test["fresh_independent_test_attested"] is False
        assert custom_test["test_observed_before_freeze"] is True
        benchmark = yaml.safe_load(
            archive.read("configs/benchmark_official_rtx2060.yaml")
        )
        assert benchmark["active_models_pointer"] == "outputs/task/active_models.yaml"
        assert benchmark["category"] == "part_a"
        assert benchmark["warmup"] == 20
        assert benchmark["runs"] == 100

        manifest = yaml.safe_load(
            archive.read(submission.MANIFEST_NAME).decode("utf-8")
        )
        assert manifest["schema_version"] == 3
        assert manifest["package_scope"] == "single_category"
        assert manifest["evidence"]["scope"] == "selected_category_only"
        assert (
            manifest["evidence"]["pooled_or_multi_category_machine_evidence_included"]
            is False
        )
        assert manifest["configuration"]["pooled_configs_included"] is False
        assert manifest["configuration"]["generated_files"] == (
            submission.PACKAGE_CONFIG_NAMES
        )
        assert manifest["datasets_excluded"] is True
        assert manifest["runtime"]["entry_point"] == submission.PACKAGE_RUNNER_NAME
        assert manifest["runtime"]["training_supported"] is True
        assert manifest["runtime"]["evaluation_supported"] is True
        assert manifest["runtime"]["feedback_supported"] is True
        assert manifest["runtime"]["active_learning_query_supported"] is True
        assert manifest["runtime"]["interactive_inference_feedback_enabled"] is True
        assert manifest["runtime"]["interactive_inference_feedback_default"] is True
        assert manifest["runtime"]["interactive_inference_no_feedback_supported"] is True
        assert manifest["runtime"]["private_evaluation_feedback_frozen"] is True
        assert set(manifest["runtime"]["bundle_dependency_files"]) >= {
            "outputs/task/run_a/model_bundle.yaml",
            "outputs/task/run_a/model.pt",
            "outputs/task/run_a/calibration.yaml",
            "outputs/task/run_a/support_receipt.yaml",
            "assets/source_model.pt",
            "assets/runtime_asset.bin",
            "assets/runtime_repo/loader.py",
        }


def test_submission_removes_private_windows_home_paths_from_support_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    run = tmp_path / "outputs" / "task" / "run_a"
    receipt_path = run / "support_receipt.yaml"
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    receipt["source_path"] = "C:" + r"\Users\ExampleUser\private\support.png"
    receipt_path.write_text(
        yaml.safe_dump(receipt, sort_keys=False), encoding="utf-8"
    )
    bundle_path = run / "model_bundle.yaml"
    bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    bundle["artifact_sha256"]["support_receipt"] = _digest(
        receipt_path.read_bytes()
    )
    bundle["bundle_hash"] = submission._payload_hash(
        {key: value for key, value in bundle.items() if key != "bundle_hash"}
    )
    bundle_path.write_text(
        yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "submission.zip"

    submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        maximum_bytes=1_000_000,
    )

    with zipfile.ZipFile(output) as archive:
        packaged = archive.read("outputs/task/run_a/support_receipt.yaml")
        assert b"C:\\Users\\" not in packaged
        portable = yaml.safe_load(packaged)
        assert portable["source_path"] == "EXTERNAL_INPUT/support.png"
        assert portable["submission_portability"][
            "absolute_workstation_paths_removed"
        ] is True
    assert submission.verify_submission_archive(output)["bundle_hash_verified"] is True


@pytest.mark.parametrize(
    ("extra_args", "feedback_enabled"),
    [((), True), (("--no-feedback",), False)],
)
def test_extracted_submission_entry_point_runs_single_image_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: tuple[str, ...],
    feedback_enabled: bool,
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "submission.zip"
    submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        maximum_bytes=1_000_000,
    )

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(output) as archive:
        archive.extractall(extracted)
    image = extracted / "input.png"
    image.write_bytes(b"fake-rgb-image")
    completed = subprocess.run(
        [
            sys.executable,
            submission.PACKAGE_RUNNER_NAME,
            "infer",
            str(image),
            "--device",
            "cpu",
            *extra_args,
        ],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = yaml.safe_load(completed.stdout)
    assert result["category"] == "part_a"
    assert result["prediction"] is False
    assert result["bytes_read"] == len(b"fake-rgb-image")
    assert result["feedback_enabled"] is feedback_enabled
    assert (result["feedback_state"] is not None) is feedback_enabled


@pytest.mark.parametrize(
    ("command", "expected"),
    [("adapt", "adapted"), ("evaluate", "evaluated")],
)
def test_extracted_submission_runs_private_task_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    expected: str,
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "submission.zip"
    submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        maximum_bytes=1_000_000,
    )

    extracted = tmp_path / f"extracted-{command}"
    with zipfile.ZipFile(output) as archive:
        archive.extractall(extracted)
    marker = extracted / f"{command}.txt"
    config = extracted / f"{command}.yaml"
    config.write_text(
        yaml.safe_dump({"marker": str(marker)}, sort_keys=False),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, submission.PACKAGE_RUNNER_NAME, command, str(config)],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == expected


def test_extracted_submission_rejects_feedback_during_private_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "submission.zip"
    submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        maximum_bytes=1_000_000,
    )

    extracted = tmp_path / "extracted-evaluate-frozen"
    with zipfile.ZipFile(output) as archive:
        archive.extractall(extracted)
    config = extracted / "evaluate-with-feedback.yaml"
    config.write_text(
        yaml.safe_dump(
            {"marker": str(extracted / "should-not-exist.txt"), "feedback_enabled": True},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, submission.PACKAGE_RUNNER_NAME, "evaluate", str(config)],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "permanently feedback-frozen" in completed.stderr


def test_verify_submission_detects_file_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "submission.zip"
    submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        maximum_bytes=1_000_000,
    )
    assert submission.verify_submission_archive(output)["files_verified"] > 0

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            value = source.read(info.filename)
            if info.filename == "outputs/task/run_a/model.pt":
                value = b"tampered"
            target.writestr(info, value)

    with pytest.raises(ValueError, match="package (byte-count|SHA-256) mismatch"):
        submission.verify_submission_archive(tampered)


def test_verify_submission_checks_bundle_after_outer_receipt_is_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "submission.zip"
    submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        maximum_bytes=1_000_000,
    )

    with zipfile.ZipFile(output) as source:
        contents = {
            info.filename: source.read(info.filename) for info in source.infolist()
        }
    artifact_name = "outputs/task/run_a/model.pt"
    contents[artifact_name] = b"tampered-model"
    manifest = yaml.safe_load(contents[submission.MANIFEST_NAME].decode("utf-8"))
    artifact_receipt = next(
        row for row in manifest["files"] if row["path"] == artifact_name
    )
    artifact_receipt["bytes"] = len(contents[artifact_name])
    artifact_receipt["sha256"] = _digest(contents[artifact_name])
    contents[submission.MANIFEST_NAME] = yaml.safe_dump(
        manifest, sort_keys=False
    ).encode("utf-8")

    tampered = tmp_path / "tampered_with_rewritten_receipt.zip"
    with zipfile.ZipFile(tampered, "w") as target:
        for name, value in contents.items():
            target.writestr(name, value)

    with pytest.raises(ValueError, match="model artifact SHA-256 mismatch"):
        submission.verify_submission_archive(tampered)


def test_submission_rejects_pooled_machine_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    evidence = tmp_path / "outputs" / "evidence" / "metrics.yaml"
    evidence.write_text(
        yaml.safe_dump(
            {
                "images": 8050,
                "per_category": {
                    "part_a": {"images": 1000},
                    "part_b": {"images": 1000},
                },
                "selection_scope": "per_category_100_normal_plus_30_defect_support_only",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pooled or multi-category evidence"):
        submission.build_submission(
            pointer=pointer,
            category="part_a",
            output=tmp_path / "submission.zip",
            evidence=(evidence,),
            maximum_bytes=1_000_000,
        )


@pytest.mark.parametrize(
    ("filename", "content", "error"),
    [
        ("metrics.csv", "category,images\npart_a,1000\n", "structured YAML/JSON"),
        ("notes.txt", "part_a result\n", "structured YAML/JSON"),
        ("metrics.yaml", "images: 1000\n", "unscoped evidence"),
        (
            "predictions.json",
            '[{"category": "part_a", "prediction": 0}]',
            "root must be a mapping",
        ),
    ],
)
def test_submission_rejects_opaque_or_unscoped_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content: str,
    error: str,
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    evidence = tmp_path / "outputs" / "evidence" / filename
    evidence.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        submission.build_submission(
            pointer=pointer,
            category="part_a",
            output=tmp_path / "submission.zip",
            evidence=(evidence,),
            maximum_bytes=1_000_000,
        )


def test_submission_records_evidence_role_count_and_false_claim_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    evidence = tmp_path / "outputs" / "evidence" / "metrics.yaml"
    evidence.write_text(
        yaml.safe_dump(
            {
                "category": "part_a",
                "images": 1000,
                "evaluation_role": "development_diagnostic",
                "eligible_for_final_independent_claim": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "submission.zip"

    submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        evidence=(evidence,),
        maximum_bytes=1_000_000,
    )

    with zipfile.ZipFile(output) as archive:
        manifest = yaml.safe_load(archive.read(submission.MANIFEST_NAME))
    assert manifest["evidence"]["records"] == [
        {
            "path": "outputs/evidence/metrics.yaml",
            "format": "yaml",
            "category": "part_a",
            "role": "development_diagnostic",
            "count": {"field": "images", "value": 1000},
            "claim_eligible": False,
            "claim_eligibility_recorded": True,
            "claim_eligibility_field": "eligible_for_final_independent_claim",
        }
    ]


def test_submission_rejects_unattested_independent_claim_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    evidence = tmp_path / "outputs" / "evidence" / "metrics.yaml"
    evidence.write_text(
        yaml.safe_dump(
            {
                "category": "part_a",
                "images": 1000,
                "evaluation_role": "independent_test",
                "eligible_for_final_independent_claim": True,
                "fresh_independent_test_attested": False,
                "test_observed_before_freeze": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fresh post-freeze attestation"):
        submission.build_submission(
            pointer=pointer,
            category="part_a",
            output=tmp_path / "submission.zip",
            evidence=(evidence,),
            maximum_bytes=1_000_000,
        )


def test_submission_size_gate_removes_partial_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "too_large.zip"

    with pytest.raises(ValueError, match="exceeding"):
        submission.build_submission(
            pointer=pointer,
            category="part_a",
            output=output,
            maximum_bytes=10,
        )

    assert not output.exists()


@pytest.mark.parametrize("target", ["pointer", "bundle", "artifact"])
def test_submission_rejects_output_overlapping_selected_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    protected = {
        "pointer": pointer,
        "bundle": tmp_path / "outputs" / "task" / "run_a" / "model_bundle.yaml",
        "artifact": tmp_path / "outputs" / "task" / "run_a" / "model.pt",
    }[target]
    original = protected.read_bytes()

    with pytest.raises(ValueError, match="output path overlaps protected"):
        submission.build_submission(
            pointer=pointer,
            category="part_a",
            output=protected,
            maximum_bytes=1_000_000,
        )

    assert protected.read_bytes() == original


def test_verify_submission_enforces_external_archive_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "submission.zip"
    submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        maximum_bytes=1_000_000,
    )

    with pytest.raises(ValueError, match="verification limit"):
        submission.verify_submission_archive(output, maximum_bytes=10)


def test_verify_submission_rejects_excessive_compression_ratio(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "compression_bomb.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("payload.bin", b"0" * 1_000_000)

    with pytest.raises(ValueError, match="unsafe compression ratio"):
        submission.verify_submission_archive(archive_path)


@pytest.mark.parametrize("member", ["../escape.txt", "NUL.txt", "folder/file:stream"])
def test_verify_submission_rejects_unsafe_member_paths(
    tmp_path: Path, member: str
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, b"unsafe")

    with pytest.raises(ValueError, match="archive member|reserved Windows path"):
        submission.verify_submission_archive(archive_path)


def test_verify_submission_bounds_total_uncompressed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "submission.zip"
    submission.build_submission(
        pointer=pointer,
        category="part_a",
        output=output,
        maximum_bytes=1_000_000,
    )

    with pytest.raises(ValueError, match="uncompressed size exceeds"):
        submission.verify_submission_archive(
            output,
            maximum_uncompressed_bytes=100,
        )


def test_submission_rejects_pointer_bundle_target_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    bundle_path = tmp_path / "outputs" / "task" / "run_a" / "model_bundle.yaml"
    bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    bundle["target_object"] = "part_b"
    bundle["bundle_hash"] = submission._payload_hash(
        {key: value for key, value in bundle.items() if key != "bundle_hash"}
    )
    bundle_path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="pointer and model bundle target disagree"):
        submission.build_submission(
            pointer=pointer,
            category="part_a",
            output=tmp_path / "submission.zip",
            maximum_bytes=1_000_000,
        )


def test_submission_rejects_operator_feedback_history_as_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = _project(tmp_path)
    monkeypatch.setattr(submission, "PROJECT_ROOT", tmp_path)
    feedback_receipt = (
        tmp_path / "outputs" / "feedback" / "revisions" / "revision-a" / "head.pt"
    )
    feedback_receipt.parent.mkdir(parents=True)
    feedback_receipt.write_bytes(b"operator-feedback-only")

    with pytest.raises(ValueError, match="operator feedback history/images/revisions"):
        submission.build_submission(
            pointer=pointer,
            category="part_a",
            output=tmp_path / "submission.zip",
            evidence=(feedback_receipt,),
            maximum_bytes=1_000_000,
        )
