<!--THE README INSTRUCTION FILE FOR JBG classification with an additional Jupyter GUI.-->
# JBG automatic spotchecking of classifiers

## Summary
This project aims on collecting many different data classifiers from several different
libraries under a common Jupyter GUI with the purpose of simplifying inital testing and 
spot-checking of algorithms for a particular dataset. For the time being JBG is limited
to a Windows and SQL Server environment and works for numerical, categorical and textual
data. It is partly a Building AI course project.

## Development records

- `CHANGELOG.md` is the append-only history of numbered development revisions and their verification status.
- `BACKLOG.md` tracks open issues, priorities and deferred work.

## How to use JBG
To use the Jupyter GUI for JBG Python autoclassification script, do as follows:

1. Get a hold of the JBGclassification GUI codes in exactly one of the two following ways:
    a) Use GIT:
        i) Install GIT, see https://github.com/git-guides/install-git 
        ii) Clone the code from the repository in a terminal window: 
                git clone https://github.com/slimebob1975/JBGautoclass-jupyter.git
            into an appropriate directory
    b) Ask a friend to get a copy of the code

    There are two versions of this. The legacy (if wanted) is tagged v1.0, and the main branch is version 2.

2. Copy the file example.env to a new file called .env and change
       the variables such that they reflect your SQL Server environment
       
3. Download and install Anaconda from: https://www.anaconda.com/products/distribution

4. Start Anaconda Navigator and launch Jupyter-lab from within

5. In the file explorer window to the left in Jupyter-lab, browse your way to the file
    JBGclassification_GUI.ipynb
6. If you see the JBG logo and some webb-like widgets in the right hand side, then all is ok!
    
Notice:
* The connection to SQL Server uses integrated security, so you will only be able to see the
databases and datatables that also show up in e.g. Microsoft SQL Server Management Studio
    
Troubleshooting:
* If you only see text in the right hand side window of Jupyter-lab, try to restart the kernel
(push the double play symbol)

== Terminal ==

To run the script in the terminal you need to have a file in `src\JBGclassification\config` with a functional config. 
Configs start with `autoclassconfig_` as the name, and will be saved when you create a new model using the GUI.
Generated config files deliberately do **not** contain the SQL password. Supply it at runtime through the
`JBG_SQL_PASSWORD` process environment variable (or enter it in the GUI). Saved `.sav` model files likewise omit
the SQL password; when a model is loaded through the application, the current runtime credentials are injected.

Revisions 084-088 provide a validation runner that reuses the most recent manual/Repeat Last configuration and compares the hard/direct baseline with `perturbed_same_model`; revision 088 adds robustness summaries and uses nine paired seeds by default. Revision 089 exposes that same shadow-clone method in the normal Dark Number path, revision 091 presents its correction-failure policy as a two-option radio choice, revisions 092-093 add/refine a live equation card, and revision 095 adds a dynamic `Target:` radio group. `All classes` preserves historical behavior; selecting one observed class restricts correction-factor estimation, fallback and Dark Number output to that target while the full confusion matrix remains available for model diagnosis. The compact failure labels are `No fallback` / `Experimental` with the experimental choice selected by default. The card is driven by `JBGDarkNumbers.py`, updates with Method + Alpha + Target + correction-failure policy, and dims when `Estimate` is off. Direct correction still wins whenever it is estimable; the experimental path is attempted only for recovery-level statistical failures and remains explicitly marked with `corr_source=perturbed_same_model`. Run the validation harness from the repository root after a model-training run:

```text
python .\src\JBGclassification\JBGDarkNumberValidationRunner.py --sql-username <username> --runs 9
```

The runner reads `.jbg_last_run.json`, uses `JBG_SQL_PASSWORD` if set (otherwise it prompts without echo), reloads the
saved model's fitted text/category converter, normalizes sparse text features to SciPy CSR, and writes paired
baseline-vs-`perturbed_same_model` CSV/JSON evidence under `src\JBGclassification\output\csvs`. Its log is
written separately as `jbg-dark-number-validation_*.log`. The perturbation defaults are five shadow clones, at least three valid clone estimates, and maximum correction-factor CV 0.50; these can be adjusted with `--perturbation-clones`, `--perturbation-min-valid`, and `--perturbation-max-cv`. Normal production runs keep those validated 5/3/0.50 guardrails fixed for now and, whenever the experimental fallback is attempted, write aggregate provenance to `dark_number_experimental_fallback_*` without raw source rows.

Revision 110 displays a mathematical four-point range directly below the Dark Number Calculations table: `[D_re_full, D_cv_full, D_comb, D_cv_test] = [values]`, separately for each target/formula. The short names refer to retrained-full, cross-trained-full, combined legacy full-data and cross-trained holdout estimates. Values use the table's proportion units, with readable logging alongside the notebook expression. The expected non-decreasing order permits ties; inversions are explicitly signalled without sorting them away. Missing or unestimable estimates produce an incomplete check, and valid zero-FP results identify their skipped correction. This is a heuristic range, not a statistical confidence interval. The calculations and CSVs remain unchanged.

Revision 108 avoids correction fits that cannot change the reported Dark Number. Confusion-matrix **rows are actual labels and columns are predicted labels**: for target `Ja`, `Nej → Ja` is FP and `Ja → Nej` is FN. In every current formula, FP=0 with observed negative support makes the false-positive multiplier zero, so the reported Dark Number remains zero for any valid finite correction factor—even if there are false negatives. That is a property of the formula, not proof that no unobserved positives exist.

The normal path caches its existing predictions once and skips correction estimation for a target/model only if every output using that correction has exactly zero FP and valid target/rest support (plus aligned complete labels and valid confidence values for alpha formulas). This includes the holdout, full-data and combined legacy reports under their existing correction ownership: a single FP in any dependent report retains correction estimation. Direct estimation, resource regression and experimental shadow clones are all omitted for unused corrections. Matrices, alphas and Dark Number rows are retained. Output labels the provenance `corr_source=not_needed_zero_fp`, with `corr=1.0` solely as a neutral calculation placeholder; it is not an estimated correction. The log reports skipped work and per-scope FP/FN counts, and warns when zero formula output coexists with false negatives. A zero-FP run consequently has no shadow-fit progress bar. Standalone correction-validation/sensitivity studies continue estimating corrections for their experimental purpose.

Revision 107 adds **fit-level progress and bounded parallel execution** to the normal experimental Dark Number fallback. Its startup log reports `5 shadow clones x folds/repeats = total fits` and the actual initial worker count. Independent sklearn fits use up to eight process workers, bounded by the available CPUs and positive `DEFAULT_N_JOBS_DESIRED` cap; `-1` still means no additional global cap. Native numerical threads are limited to one per fit and explicit nested `n_jobs` settings are capped to one. The five clones are processed in order, with each clone’s fold/repeat fits running concurrently. Existing bootstrap/flip seeds, all five clones, hard predictions, minimum three usable corrections and maximum CV 0.50 remain intact; no early acceptance or reduced sampling is introduced. Framework/checkpoint estimators stay sequential and retain the progress meter. Older joblib without result streaming uses sequential progress; no dependency upgrade is required.

The inline **Shadow fits** meter advances as fit results return in the parent kernel. It accounts for completed fits and explicitly skipped slots when a clone fails; 100% means the planned work is accounted for, not that the correction was accepted. It can pause while a slow fit is running and is not a time countdown. The log reports each clone’s returned fits, correction/status, worker count and duration, plus a final completed/skipped/elapsed summary. Process-resource retries reduce workers and retain already returned results; serialization failures retry unfinished fits sequentially. Large numeric arrays can use joblib memory mapping. The existing `dark_number_experimental_fallback_*` CSV adds planned/completed/skipped fits, initial/final workers and execution seconds. Wall-clock gains depend on fit cost, available memory and CPU contention; framework and very cheap fits may benefit little from parallelism. Stop Voilà and its training kernels before applying code patches, then restart.

Revision 103 clarifies model-selection scoring. `F1 Micro` pools class counts and equals accuracy for single-label classification over all classes; `F1 Macro` gives every class equal weight, while `F1 Weighted` weights each class by support. Hover over `Score metric` for an explanation of the current selection. For rare operational classes, consider `F1 Macro` or `Matthews Corr. Coefficient` (MCC), and review the per-class report rather than relying on aggregate accuracy. Existing models/settings and selected scoring objectives are preserved.

The evaluation information table now also reports **majority-class baseline accuracy (evaluation majority)**, **balanced accuracy** and **MCC**. The baseline is the largest class's share in that evaluation sample: the retrospective accuracy of always predicting that evaluation-majority class. It is a descriptive comparator, not a trained dummy model or a metric used to pick winners. Balanced accuracy averages recall over observed true classes; MCC uses the full confusion matrix and is 0 for constant predictions with no measured correlation. Candidate selection and GridSearch still use the scoring metric you select, using training CV rather than these held-out diagnostics.

Revision 104 displays an **approximate final GridSearch wall-clock budget** before the search starts, both in the progress label and application log. For example, `Grid search estimated wall-clock time: ~2 h 35 min; 6 parameter combinations x 10 folds = 60 CV fits + 1 refit.` The estimate uses the selected pipeline's measured CV fit/scoring durations, concurrent fit batches, an observed startup/dispatch allowance and one sequential final refit. Search worker caps are resolved, but no speedup beyond the concurrency observed in CV is assumed. The log states the assumptions: comparable cost across parameter combinations and approximately linear refit scaling. Slow hyperparameter settings, internal ensemble/RFE work, contention and retries can change the actual duration substantially. Missing valid timing produces an explicit unavailable message and training continues. This is a forecast, not a countdown or a confirmation prompt; it adds no benchmark training and keeps model selection/holdout handling intact.

Revision 106 adds a completion comparison for successful GridSearch: estimated and actual duration, `actual/estimate` percentage and signed deviation. Percentages use unrounded seconds, so rounding the displayed forecast to whole minutes does not affect the calculation. Negative deviation means faster than estimated; positive means slower. The measured interval covers CV fits and final refit. Missing forecasts produce an unavailable comparison, and failed searches/ordinary-fit fallback produce no successful GridSearch percentage.

To inspect a trained model, enable **Feature importance** in the **Feature analysis** section before a fresh training run. It defaults off; **Repeats** defaults to five and can be set from two to thirty. After final evaluation and before full-data retraining, the application shuffles each original input column on the entire holdout and measures the decrease in the selected score (for example MCC). The fitted text/category converter and full pipeline stay fixed, so original source names remain meaningful through text expansion and PCA/TSVD. Text fields are shuffled as whole fields. This works across classifier families and does not require estimator coefficients, an auxiliary logistic model, `predict_proba` for ordinary classification scorers, or extra fitting. Scorers retain their normal response requirements, such as probabilities for multiclass ROC AUC.

The output ranks the top twenty inputs by mean score decrease, reports standard deviation across shuffles and links two CSVs: `feature_importance_*` contains every feature, mean/std and raw repeat scores; `feature_importance_details_*` contains the model, scorer, baseline, row/class counts, repeat count, seed (42) and timing. A positive decrease means the fixed model relied on that input; negative values mean the shuffle improved its score and are retained. Standard deviation is repeat variation, not a confidence interval. Correlated inputs can hide each other's importance, and a weak model can assign low importance to useful data. Treat this as model inspection rather than causality or automatic feature selection. If you use the report to tune features, assess the revised model on new untouched test evidence.

Analysis adds `1 + number_of_original_inputs * repeats` score evaluations and preserves all holdout rows to retain rare classes. Execution is sequential and sparse conversion stays sparse at prediction boundaries. Optional analysis failures warn and let the normal training/retraining lifecycle continue. The option/repeat count survive saved settings and Repeat Last; older settings default off/five repeats. Prediction-only runs and regression suites omit analysis. New generated configs also fix NgramRange quoting and accept omitted mail settings on reload; older generated files containing bare names such as `UNI_GRAM` need regeneration or manual quotation before import.

Revision 099 adds a separate correction-noise sensitivity runner. It fetches the real dataset once, freezes one deterministic train/test split and one cross-trained model, then compares 5%, 10%, 15%, and 20% correction label flips across nine correction seeds by default. It records hard flipped/recovered counts, recovery/corr stability, fallback activation and configured-formula `D_cv_full`, plus dataset/split fingerprints. Run it after an ordinary training run with the operational Dark Number target selected:

```text
python .\src\JBGclassification\JBGDarkNumberNoiseSensitivityRunner.py --sql-username <username>
```

Use repeated `--fraction` or `--target` arguments to override the default grid/target. If the configured Dark Number target is unset and no `--target` is supplied, all observed classes are studied, so the total cell count grows by the number of targets. Revision 100 logs the resolved targets, the loaded and fixed pipeline identities, and `[n/N]` progress for every completed fraction×seed cell; the metadata JSON also retains full source/fixed estimator identity. The command is validation-only and does not change the production 20% default. Current sensitivity evidence keeps 20% as the compatibility default while 15% is tracked as a lower-perturbation candidate pending another genuinely imbalanced real dataset.

Revision 102 automatically checkpoints the sensitivity study after each completed target×fraction×seed cell. The default checkpoint is `src\JBGclassification\output\csvs\dark_number_noise_sensitivity_<model>_checkpoint.json`; its `.model.joblib` companion preserves the **original fitted fixed pipeline and predictions**, while `.inputs.joblib` and `.inputs.json` preserve/checksum the original fetched feature/label/source-pipeline snapshot. Keep all checkpoint companions together. Resuming uses that snapshot without a new SQL fetch or application shuffle; this matters because both can randomize the row selection/order. Metadata/fingerprints are written before fixed-model training begins. Completed cells survive interruption, including a forced process close, and the interrupted cell is rerun when you issue the same command again. The log reports `Checkpoint progress: n/N completed cells`; resuming restores the fixed model without retraining it. A complete checkpoint regenerates the final CSV/summary/metadata outputs without repeating the experiment.

Resume validates the ordered dataset (including feature names/types), deterministic split, full model hyperparameters, saved model artifact checksum and last-run settings fingerprint, targets/fractions/seeds, calculation/fallback options, perturbation guardrails, source code and dependency versions. A mismatch stops the run without overwriting the checkpoint. For a separate fresh experiment, use a new checkpoint filename:

```powershell
python .\src\JBGclassification\JBGDarkNumberNoiseSensitivityRunner.py --target 1 --checkpoint .\creditcard_fraud_sensitivity_102.json
```

Checkpoint resume also requires the original source code and dependency versions. Applying a later code revision such as 103 intentionally invalidates a 102 checkpoint for resume; its completed CSV/metadata exports remain usable. After interruption, repeat that exact command with the same saved source model and last-run settings. Avoid retraining/replacing the ordinary source model between attempts. Each checkpoint is locked against concurrent writers; the operating system releases the lock on process exit. A resumed study measures the original data snapshot, even if the live SQL table subsequently changes; use a new checkpoint path to study fresh data. Final CSVs still contain the full study only; the checkpoint holds partial results and preserves NaN/infinite diagnostics exactly. Earlier revision-100 log-only cells cannot be resumed, so the first revision-102 study starts from cell 1. Production Dark Number defaults remain unchanged.

Go into `src\JBGclassification` and run `python JBGautomaticClassifier.py -f <path-to-file>`. The path to the file needs to
be on the format of `.config\filename.py`, so assuming that the config-file is `autoclassconfig_iris_abc0123.py` 
(check the `config` directory for the right name), the command is: `python JBGautomaticClassifier.py -f autoclassconfig_iris_abc0123.py`


Serialized `.sav` model files use Python pickle/dill-compatible serialization. New model files are written with
an explicit JBG artifact format marker and schema version; the loader also accepts the legacy six-item payload used
by earlier releases. The supported persistence contract is save/reload/predict/retrain within a compatible JBG/Python/
ML dependency environment, not portability across arbitrary Python or scikit-learn versions. Only load model files
from a trusted source: loading an untrusted pickle/dill payload can execute arbitrary code.

=== Troubleshooting ===

1. You have to run the terminal command from the src\JBGclassification directory, due to imports and such
2. There are a lot of (unnecessary) warnings coming out of the 3rd-party libraries (in particular sklearn), which will clutter up
the terminal. To ignore them, use the `W` flag in the command (see below for usage)

==== W-flag ====

Source: https://docs.python.org/3/using/cmdline.html#cmdoption-W

To ignore all warnings: `python -Wi JBGautomaticClassifier.py -f <path-to-file>`

The full argument is `action:message:category:module:lineno`, and if you're targetting something "deeper" into the argument, leave any
intermediate things empty, IE `ignore::Classname` to target all warnings of Classname, no matter their message.

You can also put in as many of these specific warning-suppressions as you want, where for ease of use I'll write down the fine-grained
"most common" warnings below.

Specific warnings we know are irrelevant:
* UserWarning (-Wi::UserWarning)
* RuntimeWarning (-Wi::RuntimeWarning))

## Acknowledgments
This work was partly inspired by Jason Brownlee: https://machinelearningmastery.com/


### Dark Number direct-correction diagnostics

Revision 097 orders the controls as `Estimate -> Target -> Method -> Alpha -> Failure` and warns when a direct correction factor is based on mean recovery below 5% (`corr > 20`). The value remains a direct estimate; the application does not clamp it or trigger fallback solely because of the warning.
