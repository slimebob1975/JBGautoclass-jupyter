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

Revision 099 adds a separate correction-noise sensitivity runner. It fetches the real dataset once, freezes one deterministic train/test split and one cross-trained model, then compares 5%, 10%, 15%, and 20% correction label flips across nine correction seeds by default. It records hard flipped/recovered counts, recovery/corr stability, fallback activation and configured-formula `D_cv_full`, plus dataset/split fingerprints. Run it after an ordinary training run with the operational Dark Number target selected:

```text
python .\src\JBGclassification\JBGDarkNumberNoiseSensitivityRunner.py --sql-username <username>
```

Use repeated `--fraction` or `--target` arguments to override the default grid/target. If the configured Dark Number target is unset and no `--target` is supplied, all observed classes are studied, so the total cell count grows by the number of targets. Revision 100 logs the resolved targets, the loaded and fixed pipeline identities, and `[n/N]` progress for every completed fraction×seed cell; the metadata JSON also retains full source/fixed estimator identity. The command is validation-only and does not change the production 20% default. Current sensitivity evidence keeps 20% as the compatibility default while 15% is tracked as a lower-perturbation candidate pending another genuinely imbalanced real dataset.

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
