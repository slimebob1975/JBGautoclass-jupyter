"""Atomic, process-locked checkpoints for the validation-only sensitivity study."""

from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import joblib


CHECKPOINT_VERSION = 1


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity() -> dict:
    """Refuse to mix results produced by different code/dependency environments."""
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib",
                 "imbalanced-learn", "tensorflow", "scikeras", "torch", "jax"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": list(sys.version_info[:3]),
        "packages": packages,
        "code": {path.name: file_fingerprint(path)
                 for path in sorted(Path(__file__).parent.glob("*.py"))},
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": str(value)}
    return value


def _json_restore(value):
    if isinstance(value, dict):
        if set(value) == {"__float__"} and value["__float__"] in {"nan", "inf", "-inf"}:
            return float(value["__float__"])
        return {key: _json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    return value


def _atomic_write(path: Path, writer):
    """Replace only after a complete, flushed write on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_input_snapshot(path: Path, request: dict):
    """Restore the original SQL selection/order instead of making a fresh random draw."""
    manifest_path = path.with_name(path.name + ".inputs.json")
    input_path = path.with_name(path.name + ".inputs.joblib")
    if not manifest_path.exists():
        if path.exists():
            raise ValueError(f"Sensitivity input checkpoint is missing: {manifest_path}")
        return None
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (ValueError, OSError) as error:
        raise ValueError(f"Cannot read sensitivity input checkpoint: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Invalid sensitivity input checkpoint: {manifest_path}")
    if manifest.get("request") != _json_safe(request):
        raise ValueError(
            "Sensitivity input checkpoint mismatch: last-run settings, source model or "
            "code/dependencies changed. Use a new --checkpoint path for a new study."
        )
    if not input_path.is_file() or file_fingerprint(input_path) != manifest.get("sha256"):
        raise ValueError(f"Sensitivity input checkpoint is missing or damaged: {input_path}")
    return joblib.load(input_path)


def save_input_snapshot(path: Path, request: dict, bundle: dict):
    input_path = path.with_name(path.name + ".inputs.joblib")
    manifest_path = path.with_name(path.name + ".inputs.json")
    _atomic_write(input_path, lambda handle: joblib.dump(bundle, handle))
    manifest = {"version": CHECKPOINT_VERSION, "request": request,
                "sha256": file_fingerprint(input_path)}
    data = json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _atomic_write(manifest_path, lambda handle: handle.write(data))


@contextmanager
def checkpoint_lock(path: Path | None):
    """An OS lock is released on process exit, including forced Windows logout."""
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"Sensitivity checkpoint is in use by another process: {path}") from error
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    # Keep the lock file: unlinking it can let competing processes lock different inodes.


class SensitivityCheckpoint:
    def __init__(self, path: Path, experiment: dict, planned_cells: list[tuple], metadata: dict | None = None):
        self.path = path
        self.model_path = path.with_name(path.name + ".model.joblib")
        self.planned_cells = planned_cells
        if len(set(planned_cells)) != len(planned_cells):
            raise ValueError("Sensitivity targets and fractions must not contain duplicates.")
        self.rows = []
        self.payload = {
            "version": CHECKPOINT_VERSION,
            "experiment": experiment,
            "metadata": metadata or {},
            "state": "prepared",
            "model_sha256": None,
            "rows": [],
            "completed_cells": 0,
            "total_cells": len(planned_cells),
        }
        if path.exists():
            try:
                with path.open(encoding="utf-8") as handle:
                    saved = json.load(handle)
            except (ValueError, OSError) as error:
                raise ValueError(f"Cannot read sensitivity checkpoint {path}; it was not overwritten.") from error
            if not isinstance(saved, dict) or saved.get("version") != CHECKPOINT_VERSION:
                raise ValueError(f"Unsupported sensitivity checkpoint: {path}")
            if saved.get("experiment") != _json_safe(experiment):
                raise ValueError(
                    f"Sensitivity checkpoint mismatch: {path}. Dataset, split, model, settings or "
                    "code/dependencies changed. Use a new --checkpoint path for a new study."
                )
            rows = saved.get("rows")
            if (not isinstance(rows, list) or len(rows) > len(planned_cells)
                    or saved.get("completed_cells") != len(rows)
                    or saved.get("total_cells") != len(planned_cells)
                    or saved.get("state") not in {"prepared", "running", "complete"}):
                raise ValueError(f"Invalid sensitivity checkpoint progress: {path}")
            for index, row in enumerate(rows):
                if not isinstance(row, dict) or self.cell_key(row) != planned_cells[index]:
                    raise ValueError(f"Invalid/duplicate sensitivity checkpoint cell: {path}")
            if ((saved["state"] == "prepared" and (rows or saved.get("model_sha256")))
                    or (saved["state"] != "prepared" and not saved.get("model_sha256"))
                    or (saved["state"] == "complete" and len(rows) != len(planned_cells))):
                raise ValueError(f"Invalid sensitivity checkpoint state: {path}")
            self.payload = saved
            self.rows = _json_restore(rows)
        else:
            self.write()

    @staticmethod
    def cell_key(row: dict) -> tuple:
        return row.get("target"), row.get("flip_fraction"), row.get("correction_seed")

    def write(self):
        payload = dict(self.payload, rows=self.rows, completed_cells=len(self.rows))
        data = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        _atomic_write(self.path, lambda handle: handle.write(data))

    def load_fixed_experiment(self):
        expected = self.payload.get("model_sha256")
        if expected is None:
            return None
        if not self.model_path.is_file() or file_fingerprint(self.model_path) != expected:
            raise ValueError(f"Sensitivity fixed-model checkpoint is missing or damaged: {self.model_path}")
        return joblib.load(self.model_path)

    def save_fixed_experiment(self, bundle: dict, metadata: dict):
        _atomic_write(self.model_path, lambda handle: joblib.dump(bundle, handle))
        self.payload.update(model_sha256=file_fingerprint(self.model_path), metadata=metadata, state="running")
        self.write()

    def append(self, row: dict):
        if (len(self.rows) >= len(self.planned_cells)
                or self.cell_key(row) != self.planned_cells[len(self.rows)]):
            raise ValueError("Sensitivity cell does not match the next planned checkpoint cell.")
        self.rows.append(row)
        self.write()

    def finish(self):
        if len(self.rows) != len(self.planned_cells):
            raise ValueError("Cannot complete an unfinished sensitivity checkpoint.")
        self.payload["state"] = "complete"
        self.write()
