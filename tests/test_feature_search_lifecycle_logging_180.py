"""Rev180: lifecycle logging is privacy-minimal and does not need Voilà."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def _method(name):
    source = Path(__file__).resolve().parents[1] / 'src' / 'GUI' / 'Widgets.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                  and cls.name == 'Widgets' for node in cls.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {'logging': logging}
    exec(compile(module, str(source), 'exec'), scope)
    return scope[name]


def test_lifecycle_log_does_not_expose_callback_metadata(caplog):
    handler = _method('_feature_search_event')
    messages = []
    fake = SimpleNamespace(guihandler=SimpleNamespace(logger=SimpleNamespace(
        log_file=SimpleNamespace(write_message=messages.append))))
    fake._feature_search_lifecycle_log = lambda *args: _method('_feature_search_lifecycle_log')(fake, *args)
    secret = 'SENSITIVE_SOURCE_180'
    for name in ('study_started', 'study_resumed', 'study_paused',
                     'study_interrupted', 'study_failed', 'study_completed'):
        handler(fake, {'event': name, 'completed_candidates': 4, 'attempted_fits': 12,
                       'study_id': secret, 'error': secret, 'model_type': secret,
                       'output_directory': secret})
    assert len(messages) == 6
    assert all('completed_candidates=4 attempted_fits=12' in line for line in messages)
    assert secret not in '\n'.join(messages)


def test_pause_request_log_and_event_are_idempotent(caplog):
    handler = _method('pause_feature_selection_investigation')
    event = Mock()
    messages = []
    fake = SimpleNamespace(_feature_search_running=True, _feature_search_pause_event=event,
                           feature_search_pause_button=SimpleNamespace(disabled=False),
                           feature_search_status=SimpleNamespace(value=''),
                           guihandler=SimpleNamespace(logger=SimpleNamespace(log_file=SimpleNamespace(write_message=messages.append))))
    fake._feature_search_lifecycle_log = lambda *args: _method('_feature_search_lifecycle_log')(fake, *args)
    with caplog.at_level(logging.INFO):
        handler(fake)
    event.set.assert_called_once_with()
    assert fake.feature_search_pause_button.disabled
    assert 'event=pause_requested' in messages[0]
