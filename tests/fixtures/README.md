# Test fixtures

Serialized `.sav` model files in this directory are generated local test artifacts and are intentionally not version-controlled. They are pickle/dill-based binaries, so they are dependency-version-sensitive, opaque to review, and must only be loaded when their origin is trusted.

Use `src/JBGclassification/regenerate-fixtures.py` when the legacy persistence tests need `config-save.sav` or `model-save.sav`. The generated files stay local and are ignored by Git.

If a future compatibility test genuinely requires a frozen serialized model, document the exact producer/runtime version and explicitly whitelist that specific fixture instead of committing arbitrary generated `.sav` files.
