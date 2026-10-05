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

### PyTorch variants and training profile (116–117)

All six named variants remain available: TORA/TORS use ReLU, TOTA/TOTS use Tanh, and TOSA/TOSS use Sigmoid; the final A/S selects Adam/SGD. The selected activation and optimizer now remain fixed during that variant's final search. Adam is passed to skorch explicitly rather than silently training its variants with the default SGD. Inference disables dropout and repeated probability predictions are deterministic for a fixed fitted CPU model.

| Variants | Initial learning rate | Search learning rates | Hidden stages | Widths |
| --- | --- | --- | --- | --- |
| TORA/TOTA/TOSA (Adam) | 0.001 | 0.0003, 0.001, 0.003 | 3 | 48, 100 |
| TORS/TOTS (ReLU/Tanh + SGD) | 0.02 | 0.01, 0.02, 0.05 | 3 | 48, 100 |
| TOSS (Sigmoid + SGD) | 0.1 | 0.02, 0.05, 0.1 | 1 | 48, 100 |

Each named search has six combinations (60 CV fits plus one refit with ten folds), replacing the old shared 1944-combination search (19440 CV fits plus one refit). The profile fixes 50 epochs, batch size 128, dropout 0.1 and skorch's internal stratified validation split. TOSS uses zero **additional** hidden stages after the first input-to-hidden layer (one hidden stage in total); the other five variants use two additional stages (three in total). The constructor's depth interpretation and its default of two additional stages remain compatible with existing code/artifacts. It still accepts custom epochs/depth/dropout and `train_split=False`; the generic GUI iteration setting does not set PyTorch epochs. Widths/rates and the 50-epoch budget are bounded starting choices rather than evidence of convergence or optimal accuracy on every dataset. Validate them on appropriately scaled data and inspect minority-class performance.

Revision 117 responds to the weak scaled Sigmoid+SGD results in the 116 Breast Cancer run. A paired CPU study of depth and learning rate found that the shallower profile improved AUC in all 18 fold/seed/preprocessing comparisons, with improved malignant-class recall. This supports a bounded TOSS-specific starting profile, not a universal choice or a claim about every reduction/dataset. See [the measurements and reproducible study](docs/verification/117_pytorch_sigmoid_sgd.md). Existing fitted models keep their stored architecture/rate when loaded; retrain to use the new profile. The runtime grid getter preserves revision-116 enum values for serialized artifacts.

Each fit/fold/refit/retry writes checkpoints in its own temporary subdirectory under `output/nn_checkpoints`, restores the chosen weights before deleting its temporary files, and preserves unrelated historical files. The callback restores the best validation-loss checkpoint (training-loss checkpoint without internal validation); despite its historical helper name it does **not** perform early stopping. Softmax with skorch's NLLLoss probability contract is retained. Fitted prediction and saved model reload do not depend on checkpoint files. Only the skorch component is encoded with standard pickle inside the existing dill artifact to avoid dill traversing Torch optimizer internals; the historical plain-net state remains readable. Historical Algorithm/PYNN enum values and class import paths remain unchanged. Retrain to obtain the corrected Adam behavior; loading previous weights does not retrain them. Existing automatic CUDA selection and the sequential experimental framework fallback policy are retained; local validation is CPU-only and does not establish GPU memory/concurrency behavior.

API references: [skorch optimizer/probability/loss contract](https://skorch.readthedocs.io/en/stable/user/neuralnet.html), [Checkpoint versus EarlyStopping](https://skorch.readthedocs.io/en/stable/callbacks.html), and [PyTorch dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).

### GridSearch time estimates (118)

Final-search estimates apply to every model using the shared GridSearch path. A first search uses the selected pipeline's measured CV times, concurrent batches, startup/dispatch allowance and approximate sequential refit. A successful search records its actual/base duration ratio in `src/JBGclassification/output/grid_search_timing.json`. Future matching searches multiply the current CV-based estimate by the median of up to five matching observations from the last 30 days. The file holds at most 64 profiles and survives a kernel restart; delete it to reset calibration. Existing log files are not imported automatically.

Matching covers the data source/table/target, training shape/input container, feature names/types and class counts; unfitted pipeline parameters and complete search grid; scorer, fold count, resolved search workers and observed CV workers; coarse factor-of-two bands of measured CV fold work and dispatch allowance (floored at 1 ms) to avoid mixing cold/warm or materially different timing regimes; host, Python/framework versions and thread environment. This is a workload signature, not a dataset-content fingerprint: changed values in the same table, randomization, worker startup and contention can still affect duration. The history stores only hashed signatures and numeric timings, with no rows, labels, credentials or fitted models. Unsupported custom objects, missing context/telemetry, expired/corrupt history or file-access failures fall back to the CV-based estimate and cannot discard a successful model.

The log states whether an estimate is uncalibrated or learned from matching completed searches, reports the sample count and observed factor range, and keeps the actual/estimate percentage comparison. The observed range describes historical variability; it is not a confidence interval. Completion logs additionally report candidate mean fit costs, scoring, actual refit time and the remaining CV/dispatch phase. Startup/dispatch/contention are not separately measured. Calibration adds no training fits and does not change search parameters, scores, worker policy or model selection. Failed searches and ordinary-fit fallback do not contribute history. Windows verification of calibrated repeat-run forecasts remains pending.

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

Revision 111 preflight-skips Gaussian Naive Bayes when sparse features reach the estimator unchanged (NOR/RFE); dense input and PCA/TSVD/Nystroem paths remain eligible. Normal estimator calls retain sparse input and retry with dense features only for a confirmed scikit-learn sparse-X rejection, including joblib worker traceback evidence. A retry logs the operation/pipeline, shape, dtype and estimated dense-buffer size before allocation. Unrelated TypeErrors and failed dense retries are surfaced without another fit or generic serial-CV retry; existing resource/pickling worker reductions remain intact. This improves error attribution and avoids known incompatible work. Expensive Nystroem candidate fits require separate runtime investigation.

Revision 112 covers Dark Number scope predictions too. Once a model's sparse-X rejection is confirmed, its successful dense input is reused for probabilities and its dense requirement is retained for later scopes and correction fitting within that calculation. Direct corrections, regression and experimental perturbed clones receive the compatible representation, with conversion size logged before allocation. The requirement is tracked per model; sparse-capable models remain sparse and zero-FP targets still skip correction fitting and correction-input allocation. To verify the mixed-data failure fix efficiently, first select MAX preprocessing, NOR reduction and HIST only; confirm Dark Number tables and the four-point range finish before repeating the broad search.

Revision 113 adds runtime measurements to existing spot-check CV folds for NYS with MLPC, MLP2, FUTV or FUTS. Each completed candidate logs mean Nystroem fit/transform time, remaining pipeline fit time, Nystroem scoring-transform time, retained MLP iteration counts/cap hits and observed native thread counts. Fold rows, dimensions after sampling, dense feature-buffer bytes, kernel settings, scores, workers and CV wall time are exported beside the normal cross-validation CSV as `<crossval stem>_nystroem_runtime.csv`, with a download link. The residual fit time includes sampling, preprocessing and the classifier; it must not be described as MLP-only. Stacking's discarded internal fit iterations are not available. Reaching `max_iter` is reported as a cap hit, not proof of failure to converge.

Only disposable CV clones are instrumented. There are no extra fits or score calls; parameters, sampling, folds, score selection, search limits and saved model types stay unchanged. Optional metadata/export failures warn or report unavailable measurements while retaining successful CV scores. Custom Nystroem subclasses, non-default output containers and cached pipelines keep the original uninstrumented path. This revision measures the expensive paths before changing their behavior; it does not yet claim a speedup. For a targeted runtime investigation, select MAX and STA preprocessing, NYS reduction, and MLPC/FUTV (optionally FUTS), keeping dataset and iteration settings comparable; return the runtime CSV and log. Normal final training/GridSearch still follows spot-checking.

Revision 114 bounds spot-check CV's worker pool by its number of fold tasks, in addition to available CPUs and a positive configured worker cap. Previously the default `-1` worker setting accidentally replaced the fold bound with an all-CPU request: the real ten-fold profile consequently requested 24 workers. Ten tasks can only use ten workers concurrently. Resource retries still reduce that bounded pool and successful timing/diagnostics record the resulting worker count. Final GridSearch keeps its separate all-CPU policy because it usually has many more tasks. Joblib may assign a different native-thread allowance to the smaller pool; the runtime CSV records observed thread counts for verification, without promising a speedup.

The real 113 profile confirmed that Nystroem itself used only 0.04–0.44% of mean fold fit time. All four variants mapped about 20338 resampled rows into 321 dense features, and all observed native-thread counts were one. Retained MLPs used 162–605 iterations against a 20000 cap. In that run STA/NYS/FUTV took 4 min 59 s versus MAX's 27 min 40 s, and STA/NYS/MLPC took 2 min 49 s versus MAX's 32 min 44 s, with higher mean CV scores for STA. STA-only with NYS/MLPC/FUTV is therefore a practical narrower profile to try on this dataset; MAX remains selectable. Unset stochastic seeds and kernel differences prevent interpreting those observations as a general paired causal proof or forcing a new global preprocessing default.

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

Revision 097 orders the controls as `Estimate -> Target -> Method -> Alpha -> Failure` and warns when a direct correction factor is based on mean recovery below 5% (`corr > 20`). Revision 115 keeps that existing limit and makes the message more actionable: it identifies the model/target, reports the rediscovery rate and correction factor, and recommends considering another pipeline combination and investigating the correctness and quality of the training dataset. This prompts investigation rather than declaring the dataset flawed. The value remains a direct estimate; the application does not clamp it or trigger fallback solely because of the warning. The sensitivity runner's warning policy is unchanged.
