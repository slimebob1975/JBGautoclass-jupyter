"""Integration checks for the source/log move and existing local artifacts."""

import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import dill
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import Helpers
from JBGImportCompatibility import install_legacy_import_aliases
from JBGLogFile import TimestampedLogFile
from JBGModelPersistence import load_model_artifact, load_model_config
from JBGPaths import LOG_DIR, REPOSITORY_ROOT, SOURCE_DIR, relocate_saved_config_paths


def _migration_module():
    path = REPOSITORY_ROOT / "scripts" / "migrate_source_layout.py"
    spec = importlib.util.spec_from_file_location("layout_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkout(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/JBGPaths.py").touch()
    return tmp_path


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_migration_preserves_local_artifacts_and_routes_only_logs_to_root(tmp_path):
    repo = _checkout(tmp_path)
    files = {
        "model/default.sav": b"saved model",
        "model/default.sav.KERA.keras": b"keras sidecar",
        "config/autoclassconfig_local.py": b"local config",
        "output/csvs/result.csv": b"local results",
        "output/csvs/study_checkpoint.json": b"checkpoint",
        "output/csvs/study_checkpoint.model.joblib": b"checkpoint model",
        "output/csvs/study_checkpoint.inputs.joblib": b"checkpoint inputs",
        "output/grid_search_timing.json": b"timing history",
        "output/logs/application.log": b"app log",
        "output/logs/server.log": b"server log",
        "personal_notes.txt": b"other local content",
    }
    for relative, data in files.items():
        _write(repo / "src/JBGclassification" / relative, data)
    _write(repo / "src/output/logs/interim.log", b"interim log")
    (repo / "src/JBGclassification/empty-folder").mkdir()
    migration = _migration_module()
    assert migration.migrate(repo, dry_run=True) == len(files) + 1
    assert not (repo / "logs").exists()
    assert migration.migrate(repo) == len(files) + 1
    for relative, data in files.items():
        destination = repo / relative.replace("output/logs/", "logs/", 1) if relative.startswith("output/logs/") else repo / "src" / relative
        assert destination.read_bytes() == data
    assert (repo / "logs/interim.log").read_bytes() == b"interim log"
    assert (repo / "src/empty-folder").is_dir()
    assert not (repo / "src/JBGclassification").exists()
    assert not (repo / "src/output/logs").exists()
    assert migration.migrate(repo) == 0


@pytest.mark.parametrize("collision", ["different-file", "directory", "parent-file", "two-log-sources"])
def test_migration_checks_every_collision_before_moving_any_file(tmp_path, collision):
    repo = _checkout(tmp_path)
    untouched = _write(repo / "src/JBGclassification/config/first.py", b"must remain")
    old = _write(repo / "src/JBGclassification/output/logs/same.log", b"original")
    if collision == "different-file":
        _write(repo / "logs/same.log", b"other")
    elif collision == "directory":
        (repo / "logs/same.log").mkdir(parents=True)
    elif collision == "parent-file":
        _write(repo / "logs", b"a file")
    else:
        _write(repo / "src/output/logs/same.log", b"other")
    with pytest.raises(ValueError, match="collision"):
        _migration_module().migrate(repo)
    assert untouched.read_bytes() == b"must remain"
    assert old.read_bytes() == b"original"
    assert not (repo / "src/config/first.py").exists()


def test_migration_recovers_identical_copies_from_an_interrupted_move(tmp_path):
    repo = _checkout(tmp_path)
    _write(repo / "src/JBGclassification/model/saved.sav", b"identical")
    _write(repo / "src/model/saved.sav", b"identical")
    assert _migration_module().migrate(repo) == 1
    assert (repo / "src/model/saved.sav").read_bytes() == b"identical"
    assert not (repo / "src/JBGclassification").exists()


def test_migration_refuses_to_run_before_the_patch(tmp_path):
    with pytest.raises(ValueError, match="Apply patch 119"):
        _migration_module().migrate(tmp_path)


def test_failed_copy_retains_original_and_removes_only_its_partial_target(tmp_path, monkeypatch):
    repo = _checkout(tmp_path)
    source = _write(repo / "src/JBGclassification/model/saved.sav", b"original model")
    migration = _migration_module()

    def failed_copy(input_file, output, **kwargs):
        output.write(b"partial")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(migration.shutil, "copyfileobj", failed_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        migration.migrate(repo)
    assert source.read_bytes() == b"original model"
    assert not (repo / "src/model/saved.sav").exists()


def test_default_config_and_generated_config_use_the_moved_resources(tmp_path):
    from Config import Config
    config = Config()
    assert config.script_path == SOURCE_DIR
    assert config.config_path == SOURCE_DIR / "config"
    assert config.get_model_filename().parent == SOURCE_DIR / "model"
    assert config.get_output_filepath("cross_validation").parent == SOURCE_DIR / "output/csvs"
    assert config.get_classification_script_path().is_file()
    generated = tmp_path / "autoclassconfig_layout.py"
    config.save_to_file(filepath=generated)
    spec = importlib.util.spec_from_file_location("generated_layout_config", generated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = Config.load_config(module)
    assert loaded.script_path == SOURCE_DIR
    assert loaded.config_path == SOURCE_DIR / "config"


def test_default_logging_is_checkout_relative_from_an_unrelated_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert LOG_DIR == REPOSITORY_ROOT / "logs"
    log = TimestampedLogFile(filename="layout-test.log")
    try:
        assert log.path == LOG_DIR / "layout-test.log"
    finally:
        log.close()
        log.path.unlink()
    assert not (tmp_path / "logs").exists()


def test_notebook_source_assets_and_launcher_use_the_flat_layout():
    notebook = json.loads((REPOSITORY_ROOT / "JBG_SML_GUI.ipynb").read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "import src.GUIHandler as GUIHandler" in code
    assert "src.JBGclassification" not in code
    assert (SOURCE_DIR / "GUI/assets/logo.png").is_file()
    assert (SOURCE_DIR / "GUI/assets/base.css").is_file()
    assert (SOURCE_DIR / "GUI/default_settings.json").is_file()
    assert (SOURCE_DIR / "config/autoclassconfig_template.py.txt").is_file()
    assert (SOURCE_DIR / "sql/CreatePredictionTables.sql.txt").is_file()
    launcher = (REPOSITORY_ROOT / "start_application_template.ps1").read_text()
    assert "Join-Path $DevRoot 'logs'" in launcher
    assert "scripts\\migrate_source_layout.py" in launcher


@pytest.mark.parametrize("prefix", ["src", "src.JBGclassification", "JBGclassification", "src.JBGClassification", "JBGClassification"])
def test_legacy_modules_share_current_function_and_class_identity(prefix):
    install_legacy_import_aliases()
    assert importlib.import_module(prefix + ".Helpers").ensure_float64 is Helpers.ensure_float64
    from Config import Config
    from JBGMeta import Algorithm
    assert importlib.import_module(prefix + ".Config").Config is Config
    assert importlib.import_module(prefix + ".JBGMeta").Algorithm.TOSS is Algorithm.TOSS


def test_saved_config_paths_rebase_only_the_old_application_directory():
    config = SimpleNamespace(
        script_path=Path("/another-checkout/src/JBGclassification"),
        config_path=Path("/another-checkout/src/JBGclassification/config"),
        io=SimpleNamespace(model_path=r"C:\old checkout\src\JBGclassification\model"),
    )
    relocate_saved_config_paths(config)
    assert config.script_path == SOURCE_DIR
    assert config.config_path == SOURCE_DIR / "config"
    assert config.io.model_path == str(SOURCE_DIR / "model")
    custom = SimpleNamespace(script_path=Path("/custom/scripts"), config_path=Path("/custom/config"), io=SimpleNamespace(model_path="./model/"))
    relocate_saved_config_paths(custom)
    assert custom.script_path == Path("/custom/scripts")
    assert custom.config_path == Path("/custom/config")
    assert custom.io.model_path == "./model/"


@pytest.mark.parametrize("format", ["legacy", "envelope", "config-only"])
def test_config_path_relocation_through_actual_model_readers(tmp_path, format):
    from JBGModelPersistence import build_model_artifact
    config = SimpleNamespace(script_path=Path("/old/src/JBGclassification"), config_path=Path("/old/src/JBGclassification/config"))
    values = (config, None, ("NOG", "NUG", "STA", "NOR", "LRN"), None, None, 2)
    artifact = build_model_artifact(*values) if format == "envelope" else list(values) if format == "legacy" else [config]
    path = tmp_path / "saved.sav"
    with path.open("wb") as stream:
        dill.dump(artifact, stream)
    assert load_model_config(path).script_path == SOURCE_DIR
    if format != "config-only":
        assert load_model_artifact(path)[0].config_path == SOURCE_DIR / "config"


def test_old_qualified_pipeline_global_reloads_in_a_fresh_process(tmp_path):
    # A real fitted pipeline contains the old qualified helper function in its
    # serialized GLOBAL. The reader must resolve it after a process restart.
    import pickle
    from JBGModelPersistence import build_model_artifact
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])
    pipeline = Pipeline([("FLT", FunctionTransformer(Helpers.ensure_float64)), ("LRN", LogisticRegression(random_state=1))]).fit(X, y)
    artifact = build_model_artifact({}, None, ("NOG", "NUG", "STA", "NOR", "LRN"), pipeline, None, 2)
    data = pickle.dumps(artifact, protocol=0)
    assert b"cHelpers\nensure_float64\n" in data
    data = data.replace(b"cHelpers\nensure_float64\n", b"csrc.JBGclassification.Helpers\nensure_float64\n")
    path = _write(tmp_path / "old.sav", data)
    code = """
import sys
from JBGModelPersistence import load_model_artifact
import Helpers
_, _, _, pipeline, _, _ = load_model_artifact(sys.argv[1])
assert pipeline.named_steps['FLT'].func is Helpers.ensure_float64
assert pipeline.predict([[0, 0], [0, 1], [1, 0], [1, 1]]).tolist() == [0, 0, 1, 1]
pipeline.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 0, 1, 1])
assert pipeline.predict([[0, 0], [1, 1]]).tolist() == [0, 1]
print('loaded, predicted and retrained')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SOURCE_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=tmp_path, capture_output=True, text=True, check=True, env=env)
    assert result.stdout.strip() == "loaded, predicted and retrained"
