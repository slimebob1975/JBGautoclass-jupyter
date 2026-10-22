# Tests

## Running tests

Run the test suite from the repository root:

```text
pytest
```

Useful variations:

- Single file: `pytest tests/<test-file>.py`
- Single test function: `pytest tests/<test-file>.py::<test_func>`
- Single test class: `pytest tests/<test-file>.py::<test_class>`
- Single function in a test class: `pytest tests/<test-file>.py::<test_class>::<test_func>`

Test-related files:

- `tests/`
- `.coverage`
- `.coveragerc`
- `pyproject.toml`

## Documentation privacy check

Before committing documentation changes, run from the repository root:

```text
python .\scripts\check_documentation_privacy.py
```

For organization-specific person, table, database or project identifiers, create a local
`.jbg_doc_privacy_blocklist` with one literal identifier per line. The file is git-ignored and
the checker reports only the entry number, never the matched private value. The built-in patterns
are deliberately conservative, so the local blocklist and manual review remain required for names
that have no reliable syntactic signature. The default command checks the current Markdown tree only.
Before external publication, use the same local blocklist to audit reachable Git history as well:

```text
python .\scripts\check_documentation_privacy.py --history
```

Revision 166 makes the normal working-tree privacy scan Git-aware: tracked Markdown is
checked by default, while local untracked notes are included only when explicitly requested:

```powershell
python .\scripts\check_documentation_privacy.py --include-untracked
```

When a real `--history` audit is clean in the working tree but reports private blocklist
matches only in the repository-root `BACKLOG.md`/`CHANGELOG.md` history, first run the
revision-166 remediation in read-only mode:

```powershell
python .\scripts\rewrite_documentation_history.py
```

After reviewing the preflight and committing all tracked changes, apply it locally with:

```powershell
python .\scripts\rewrite_documentation_history.py --apply
python .\scripts\check_documentation_privacy.py --history
```

Apply mode creates a verified pre-rewrite Git bundle outside the repository. That bundle
contains the private historical data and must itself be kept private. Revision 166a permits
ordinary unsigned annotated tags: only tags whose target commits change are recreated, while
unaffected tags remain untouched. An affected cryptographically signed tag still stops the
rewrite rather than invalidating its signature automatically. The tool never pushes,
force-pushes, fetches, expires reflogs or runs garbage collection.

History mode is deliberately fail-closed: it requires a non-empty local private blocklist and refuses
shallow repositories. It scans annotated-tag names/messages, historical Markdown blobs and commit messages reachable from local refs,
reports commit/path/line/category metadata without printing matched values or source lines, and never
rewrites history. Run it on the authoritative full repository; if it reports findings, stop publication
and plan a backed-up history rewrite separately before re-running the audit.

## Application/build identity

Revision 175 keeps the controlled application version separate from repository state. Focused tests verify Git branch/tag/short-SHA resolution, modified-worktree marking, packaged-build environment overrides, non-Git fallback, GUI placement directly below the title, and the same identity in application-log startup lines.

## Runtime log privacy

Revision 170 keeps full SQL and row-level dataset previews out of persistent logs by default while
leaving the rich operator-visible diagnostics intact. For a trusted local debugging session only, set:

```powershell
$env:JBG_LOG_SENSITIVE_DATA = "1"
```

The opt-in affects persistent SQL and the dataset `Head` snapshot. Do not enable it when logs may be
shared externally without a separate privacy review. Feature modelling also writes a start diagnostic
showing its selected candidate-worker count and the CPU/global-cap/search-budget/memory-guard constraints
that produced it.

## Coverage

- All modules: `pytest --cov-report html:coverage --cov=./`
- One module: `pytest --cov-report html:coverage --cov=Config`

## Fixtures

Fixtures live in `tests/fixtures/`.

If `test_config.py` or `test_handler.py` fail after a deliberate serialization
change, inspect the fixtures first. The fixture regeneration helper currently
lives at `src/regenerate-fixtures.py`.

## Dark Number real-dataset validation runner

Revisions 084-087 keep fallback experiments outside production and expose a separate empirical runner. The current
runner validates `perturbed_same_model`; revision 088 adds estimable-only error/stability robustness metrics and a nine-seed default, while the earlier `soft_same_model` and `adaptive_noise_same_model` experiments remain documented rejected candidates.
After an ordinary training run has produced `.jbg_last_run.json` and the corresponding model artifact, run from
the repository root:

```text
python .\src\JBGDarkNumberValidationRunner.py --sql-username <username> --runs 9
```

Use `--target <class>` one or more times to restrict the one-vs-rest targets. The SQL password is read from
`JBG_SQL_PASSWORD` when available or requested interactively. The command creates a dedicated validation log and
paired CSV/JSON outputs; it does not modify the saved model or production Dark Number settings.

## Dark Number correction-noise sensitivity runner

Revision 099 adds a second validation-only runner that holds one fetched dataset, deterministic split and cross-trained model fixed while varying only the correction flip fraction and seed. Defaults are 5/10/15/20% and nine seeds. It writes detailed and summary CSVs plus metadata fingerprints and leaves the 20% production default unchanged.

```text
python .\src\JBGDarkNumberNoiseSensitivityRunner.py --sql-username <username>
```

Focused tests cover target/default resolution, hard flipped/recovered count reconstruction, sentinel-safe stability summaries, fingerprints, output persistence, and revision-100 source/fixed pipeline identity reporting used by long-run observability.

### 089 experimental perturbed Dark Number fallback

The production Dark Number path exposes correction-failure handling as a radio choice backed by the existing boolean config contract. Revision 093 uses compact GUI copy (`No fallback` / default `Experimental`) while preserving the same controlled-failure and experimental-perturbed semantics. Focused tests cover config default/opt-out, migration from the revision-089 checkbox, generated-config/Repeat Last persistence, dependency on the main Dark Numbers control, and routing of recovery-level direct failures into `perturbed_same_model`. Revisions 092-093 also test the live equation card and all implemented Method+Alpha branches. Revision 095 adds target-specific coverage: dynamic `All classes`/observed-class radio options, target persistence/Repeat Last, equation-card target annotation, calculator/handler target restriction, and TaskRunner propagation. The panel now uses a 72/24 control/formula split so Target/Method/Alpha/Failure remain together. The displayed branch resolution and LaTeX come from `JBGDarkNumbers.py`.

### Revision 097

Dark Number regression coverage now also verifies that the control order is `Estimate -> Target -> Method -> Alpha -> Failure`, that local settings are migrated to the same target-first ordering, that direct correction estimators retain their per-split/mean recovery diagnostics, and that mean recovery below 5% emits a warning without changing the direct correction factor or corr source.
