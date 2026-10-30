"""Sampling selection compatibility at config, snapshot and widget boundaries."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import ipywidgets
import pytest

from Config import Config
from GUI.Widgets import Widgets
from GUIHandler import GUIHandler
from JBGSampling import sampling_names
from JBGMeta import Oversampling, Undersampling


@pytest.mark.parametrize('value,expected', [
    (None, ['NOG']), ('SME', ['SME']), (Oversampling.RND, ['RND']),
    ('NOG, SME', ['NOG', 'SME']), (['SME', 'NOG', 'SME'], ['SME', 'NOG']),
    ((Oversampling.NOG, Oversampling.RND), ['NOG', 'RND']),
])
def test_legacy_and_multiple_selections(value, expected):
    assert sampling_names(value, Oversampling, Oversampling.NOG) == expected


@pytest.mark.parametrize('value', [[], (), '', ['INVALID'], ['NUG'], True])
def test_invalid_selection_is_not_silently_replaced(value):
    with pytest.raises(ValueError):
        sampling_names(value, Oversampling, Oversampling.NOG)


def test_old_mode_and_selected_model_fields():
    config = Config(mode=Config.Mode(oversampler=Oversampling.SME))
    assert config.get_oversampler_abbreviations() == ['SME']
    config.mode.oversamplers = ['NOG', 'RND']
    config.mode.undersamplers = ['NUG', 'RND']
    config.mode.validate()
    assert config.get_oversampler_abbreviations() == ['NOG', 'RND']
    assert config.mode.oversampler is Oversampling.SME
    del config.mode.oversamplers
    assert config.get_oversampler_abbreviations() == ['SME']


def test_generated_config_records_candidates(tmp_path):
    config = Config(mode=Config.Mode(oversamplers=['NOG', 'RND'], undersamplers=['NUG', 'TKL']))
    filepath = tmp_path / 'generated.json'
    config.save_to_file(filepath)
    payload = json.loads(filepath.read_text(encoding='utf-8'))
    assert payload['mode']['oversamplers'] == ['NOG', 'RND']
    assert payload['mode']['undersamplers'] == ['NUG', 'TKL']
    assert payload['mode']['oversampler'] == 'NOG'
    loaded = Config.load_config_from_json(filepath)
    assert loaded.get_oversampler_abbreviations() == ['NOG', 'RND']
    assert loaded.get_undersampler_abbreviations() == ['NUG', 'TKL']
    # Legacy scalar fallback is supported for in-memory historical modes,
    # not for incomplete schema-v2 JSON (which intentionally fails closed).
    legacy_mode = Config.Mode(oversampler=Oversampling.SME)
    legacy = Config(mode=legacy_mode)
    assert legacy.get_oversampler_abbreviations() == ['SME']


def test_snapshot_restores_all_choices_and_legacy_scalar():
    config = Config(mode=Config.Mode(oversamplers=['NOG', 'SME'], undersamplers=['NUG', 'RND']))
    params = {name: getattr(config, name) for name in ('connection', 'mode', 'io', 'debug', 'mail', 'name', 'save')}
    snapshot = GUIHandler._config_params_to_last_run_snapshot(params)
    restored = GUIHandler._last_run_snapshot_to_config_params(snapshot, 'current-user', 'current-password')
    assert restored['mode'].oversamplers == ['NOG', 'SME']
    assert restored['mode'].undersamplers == ['NUG', 'RND']
    assert 'sql_password' not in snapshot['connection']
    legacy = copy.deepcopy(snapshot)
    legacy['mode'].pop('oversamplers')
    legacy['mode'].pop('undersamplers')
    legacy['mode']['oversampler'] = 'SME'
    mode = GUIHandler._last_run_snapshot_to_config_params(legacy, 'user', 'password')['mode']
    assert sampling_names(mode.oversamplers, Oversampling, mode.oversampler) == ['SME']


def test_migrated_widgets_are_real_multiselects_with_saved_choices():
    widgets = Widgets.__new__(Widgets)
    settings = json.loads((Path(__file__).parents[1] / 'src/GUI/default_settings.json').read_text())
    widgets.default_widgets = settings['widgets']
    for name, value in [('oversampler_dropdown', 'SME'), ('undersampler_dropdown', 'TKL')]:
        widgets.default_widgets[name]['type'] = 'Dropdown'
        widgets.default_widgets[name]['params']['value'] = value
    widgets.widgets = {}
    widgets._migrate_sampling_widgets()
    assert isinstance(widgets.oversampler_dropdown, ipywidgets.SelectMultiple)
    assert widgets.oversampler_dropdown.value == ('SME',)
    assert widgets.undersampler_dropdown.value == ('TKL',)
    widgets.oversampler_dropdown.value = ('NOG', 'SME')
    widgets._migrate_sampling_widgets()
    assert widgets.oversampler_dropdown.value == ('NOG', 'SME')


def test_targeted_regression_profile_clears_manual_sampling_matrix():
    mode = Config.Mode(oversamplers=['NOG', 'SME'], undersamplers=['NUG', 'RND'])
    GUIHandler._apply_regression_suite_profile({'mode': mode}, GUIHandler.REGRESSION_SUITE_PROFILES[0])
    assert mode.oversamplers is None and mode.undersamplers is None
    assert mode.oversampler is Oversampling.RND
