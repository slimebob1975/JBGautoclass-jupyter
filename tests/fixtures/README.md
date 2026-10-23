# Test fixtures

Text/data fixtures live in this directory when a test needs stable static input.

Persistence tests deliberately create current model artifacts at test time rather than
tracking or regenerating version-sensitive `.sav` binaries. Pre-revision-176 model
artifacts are unsupported and should be retrained instead of used as compatibility fixtures.
