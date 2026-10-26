"""Prepare reuse of an existing JBG model against scoped CSV data.

This is an opt-in boundary; it never starts the SQL task runner or writes
classification results. Existing model files and JSON configs are unchanged.
"""

def prepare_csv_model_runtime(model_filename, runtime_config, repository, scope, configuration_id):
    """Return saved JBG Config and CSV row adapter after identity checks.

    The model is a trusted, locally created artifact, not arbitrary upload.
    Repository scope authorization is the caller's responsibility.
    """
    source = getattr(runtime_config, "data_source", None)
    if getattr(source, "kind", None) != "csv":
        raise ValueError("A registered CSV data source is required")
    if runtime_config.should_train():
        raise ValueError("Saved-model preparation requires prediction-only mode")

    # Uses the same model file/JSON restore path as SQL. This also checks the
    # exact saved dataset and blob IDs before anything is read from storage.
    from Config import Config
    from Storage.CSVTrainingAdapter import CSVTrainingAdapter
    saved_config = Config.load_config_from_model_file(model_filename, config=runtime_config)
    adapter = CSVTrainingAdapter(repository, scope, configuration_id, saved_config)
    return saved_config, adapter
