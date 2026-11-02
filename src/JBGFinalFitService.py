"""Final-fit orchestration for a model already selected by spot-check.

The existing ModelHandler remains the owner of error handling, diagnostics,
configuration and patchable fit/search methods.
"""
from __future__ import annotations

import time
from typing import Any
import Helpers
from JBGExceptions import ModelException


def fit_selected_model(context: Any, model: Any, dh: Any, k: int, cross_validation_filepath: str) -> Any:
    """Run the existing ordinary-fit/grid-search/fallback sequence unchanged."""
    # Now train the picked model on training data, either with a grid search or ordinary fit.
    # We assume the algorithm is the last step in the pipeline.
    model_name = model.get_name()
    alg_name = model_name.split('-')[-1]
    t0 = time.time()
    if not model.algorithm.search_params.parameters:
        context.handler.logger.print_progress(f"Using ordinary fit for final training of model {model_name}...(consider adding grid search parameters)")
        model.pipeline = context.train_picked_model(model.pipeline, dh.X_train, dh.Y_train)
    else:
        context.handler.logger.print_progress(f"Using grid search for final training of model {model_name}...")

        # Doing a grid search, we must pass on search parameters to algorithm with '__' notation.
        #search_params = \
        #    Helpers.add_missing_prefix_double_underscore_to_dict_keys(prefix=alg_name, the_dict=model.algorithm.search_params.parameters)
        search_params = \
            Helpers.add_prefix_to_dict_keys(prefix=alg_name, the_dict=model.algorithm.search_params.parameters)
        context.handler.logger.print_info(f"Using grid search parameters: {str(search_params)} for model {model_name}")
        try:
            # grid_cv_info is no used right now, but stays for debugging purposes
            model.pipeline, grid_cv_info = context.model_parameter_grid_search(
                model.pipeline,
                search_params,
                k,
                dh.X_train,
                dh.Y_train)
            context.handler.logger.print_code("Optimized parameters after grid search", model.pipeline.get_params(deep=False))
            context._export_final_search_results(grid_cv_info, cross_validation_filepath)
        except ModelException as ex:
            # In case all estimators failed to fit, use ordinary fit as fallback
            context.handler.logger.print_info(f"Grid search failed: {str(ex)}. Using ordinary fit as fallback for: {model_name}. " + \
                f"You might need to adjust grid search parameters for pipeline: {str(model.pipeline)}")
            model.pipeline = context.train_picked_model(model.pipeline, dh.X_train, dh.Y_train)
    t1 = time.time()
    context.handler.logger.print_info(f"Final training of model {alg_name} took {str(round(t1-t0,2))} secs.")
    return model
