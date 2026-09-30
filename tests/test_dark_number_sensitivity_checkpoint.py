import json

import numpy as np
import pytest

import JBGDarkNumberSensitivityCheckpoint as checkpoint


def _checkpoint(tmp_path):
    cells = [("1", 0.2, 42), ("1", 0.2, 43)]
    return checkpoint.SensitivityCheckpoint(tmp_path / "study.json", {"dataset": "abc"}, cells)


def test_atomic_write_failure_preserves_last_completed_cell(tmp_path, monkeypatch):
    study = _checkpoint(tmp_path)
    study.save_fixed_experiment({"model": "fixed"}, {"model": "fixed"})
    study.append({"target": "1", "flip_fraction": 0.2, "correction_seed": 42, "recovery": float("nan")})
    original = study.path.read_bytes()

    def failed_replace(*args):
        raise OSError("simulated disk/write failure")

    monkeypatch.setattr(checkpoint.os, "replace", failed_replace)
    with pytest.raises(OSError):
        study.append({"target": "1", "flip_fraction": 0.2, "correction_seed": 43})
    assert study.path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    restored = _checkpoint(tmp_path)
    assert len(restored.rows) == 1
    assert np.isnan(restored.rows[0]["recovery"])
    assert json.loads(original)["rows"][0]["target"] == "1"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_model_bundle_must_match_its_checksum(tmp_path, damage):
    study = _checkpoint(tmp_path)
    study.save_fixed_experiment({"model": "fixed"}, {})
    if damage == "missing":
        study.model_path.unlink()
    else:
        study.model_path.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="missing or damaged"):
        study.load_fixed_experiment()


def test_process_lock_refuses_concurrent_access_and_releases_after_exception(tmp_path):
    path = tmp_path / "study.json"
    with pytest.raises(KeyboardInterrupt):
        with checkpoint.checkpoint_lock(path):
            with pytest.raises(ValueError, match="another process"):
                with checkpoint.checkpoint_lock(path):
                    pytest.fail("Concurrent writers acquired the same checkpoint lock")
            raise KeyboardInterrupt
    with checkpoint.checkpoint_lock(path):
        pass


@pytest.mark.parametrize("damage", ["truncated", "version", "duplicate", "too_many", "count", "state"])
def test_invalid_checkpoint_is_not_overwritten(tmp_path, damage):
    study = _checkpoint(tmp_path)
    study.save_fixed_experiment({"model": "fixed"}, {})
    study.append({"target": "1", "flip_fraction": 0.2, "correction_seed": 42})
    payload = json.loads(study.path.read_text())
    if damage == "truncated":
        study.path.write_text('{"version":')
    else:
        if damage == "version":
            payload["version"] += 1
        elif damage == "duplicate":
            payload["rows"] *= 2
            payload["completed_cells"] = 2
        elif damage == "too_many":
            payload["rows"] *= 3
            payload["completed_cells"] = 3
        elif damage == "count":
            payload["completed_cells"] = 2
        elif damage == "state":
            payload["state"] = "complete"
        study.path.write_text(json.dumps(payload))
    original = study.path.read_bytes()
    with pytest.raises(ValueError):
        _checkpoint(tmp_path)
    assert study.path.read_bytes() == original


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_diagnostics_round_trip_exactly(tmp_path, value):
    study = _checkpoint(tmp_path)
    study.save_fixed_experiment({"model": "fixed"}, {})
    study.append({"target": "1", "flip_fraction": 0.2, "correction_seed": 42, "direct_corr": value})
    # JSON itself remains standards-compliant, including zero-recovery corr=inf.
    json.loads(study.path.read_text(), parse_constant=lambda value: pytest.fail(value))
    actual = _checkpoint(tmp_path).rows[0]["direct_corr"]
    if np.isnan(value):
        assert np.isnan(actual)
    else:
        assert actual == value


def test_abrupt_process_exit_keeps_cell_and_releases_lock(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    path = tmp_path / "study.json"
    program = """
import os
from pathlib import Path
import sys
from JBGDarkNumberSensitivityCheckpoint import SensitivityCheckpoint, checkpoint_lock
path = Path(sys.argv[1])
with checkpoint_lock(path):
    study = SensitivityCheckpoint(path, {'dataset': 'abc'}, [('1', 0.2, 42), ('1', 0.2, 43)])
    study.save_fixed_experiment({'model': 'fixed'}, {})
    study.append({'target': '1', 'flip_fraction': 0.2, 'correction_seed': 42})
    os._exit(77)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(checkpoint.__file__).parent)
    result = subprocess.run([sys.executable, "-c", program, str(path)], env=environment,
                            capture_output=True, timeout=30)
    assert result.returncode == 77, result.stderr.decode()
    with checkpoint.checkpoint_lock(path):
        restored = _checkpoint(tmp_path)
        assert len(restored.rows) == 1
        assert restored.load_fixed_experiment()["model"] == "fixed"
        restored.append({"target": "1", "flip_fraction": 0.2, "correction_seed": 43})
        restored.finish()
    assert json.loads(path.read_text())["state"] == "complete"


def test_input_snapshot_preserves_original_row_selection_and_order(tmp_path):
    path = tmp_path / "study.json"
    request = {"settings": "abc", "source_model": "def"}
    assert checkpoint.load_input_snapshot(path, request) is None
    bundle = {"X": np.array([[3.0], [1.0], [2.0]]), "y": np.array(["1", "0", "1"])}
    checkpoint.save_input_snapshot(path, request, bundle)
    restored = checkpoint.load_input_snapshot(path, request)
    np.testing.assert_array_equal(restored["X"], bundle["X"])
    np.testing.assert_array_equal(restored["y"], bundle["y"])


@pytest.mark.parametrize("damage", ["settings", "model", "environment", "missing", "corrupt", "manifest"])
def test_input_snapshot_refuses_changed_request_or_corruption(tmp_path, damage):
    path = tmp_path / "study.json"
    request = {"settings": "abc", "model": "def", "environment": "xyz"}
    checkpoint.save_input_snapshot(path, request, {"X": np.array([[1.0]])})
    manifest_path = path.with_name(path.name + ".inputs.json")
    input_path = path.with_name(path.name + ".inputs.joblib")
    if damage in request:
        request[damage] = "changed"
    elif damage == "missing":
        input_path.unlink()
    elif damage == "corrupt":
        input_path.write_bytes(b"broken")
    else:
        manifest_path.write_text("{truncated")
    original = manifest_path.read_bytes()
    with pytest.raises(ValueError):
        checkpoint.load_input_snapshot(path, request)
    assert manifest_path.read_bytes() == original


def test_existing_checkpoint_without_input_manifest_is_refused(tmp_path):
    study = _checkpoint(tmp_path)
    original = study.path.read_bytes()
    with pytest.raises(ValueError, match="input checkpoint is missing"):
        checkpoint.load_input_snapshot(study.path, {})
    assert study.path.read_bytes() == original
