"""Narrow, observable sparse-to-dense compatibility retry at estimator boundaries."""

import re

from scipy import sparse


SPARSE_X_ERRORS = {
    "Sparse data was passed for X, but dense data is required. "
    "Use '.toarray()' to convert to a dense numpy array.",
    "Sparse data was passed, but dense data is required. "
    "Use '.toarray()' to convert to a dense numpy array.",
    "A sparse matrix was passed, but dense data is required. "
    "Use X.toarray() to convert to a dense numpy array.",
}


def is_sparse_input_rejection(error, data):
    """Confirm sklearn's rejection of sparse X, including a joblib worker cause.

    A matching message alone is insufficient: an estimator can raise any TypeError.
    Require sklearn's validation frame locally or in a serialized worker traceback.
    Rejections of sparse y and sparse intermediates with already dense X are not
    fixed by densifying the original feature input.
    """
    if not sparse.issparse(data) or not isinstance(error, TypeError):
        return False
    if str(error) not in SPARSE_X_ERRORS:
        return False
    tb = error.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        filename = code.co_filename.replace("\\", "/")
        if filename.endswith("/sklearn/utils/validation.py") and code.co_name == "_ensure_sparse_format":
            return True
        tb = tb.tb_next
    cause = error.__cause__
    if cause is not None and type(cause).__name__ == "_RemoteTraceback":
        return bool(re.search(
            r'File "[^"\n]*[\\/]sklearn[\\/]utils[\\/]validation\.py", line \d+, in _ensure_sparse_format',
            str(cause),
        ))
    return False


def pipeline_input_label(estimator):
    steps = getattr(estimator, "steps", None)
    return "-".join(str(name) for name, _ in steps) if steps else type(estimator).__name__


def call_with_sparse_input_retry(operation, data, *, logger, context, estimator):
    """Call once normally; retry once only after a confirmed sparse-X rejection.

    Convert the actual prepared SciPy input, retaining dtype and row/column order.
    Log the single-buffer size before allocation. It is not a total worker-memory
    estimate. Resource/pickling retry policy remains with execute_n_job.
    """
    try:
        return operation(data)
    except TypeError as error:
        if not is_sparse_input_rejection(error, data):
            raise
        dense_bytes = int(data.shape[0]) * int(data.shape[1]) * int(data.dtype.itemsize)
        logger.print_warning(
            f"Confirmed sparse-X rejection during {context}, pipeline {pipeline_input_label(estimator)}: "
            f"input={type(data).__name__}, shape={data.shape}, dtype={data.dtype}, "
            f"estimated dense buffer={dense_bytes} bytes ({dense_bytes / 1024**2:.2f} MiB). "
            "CV workers and estimator copies can require additional memory. "
            f"Retrying once with dense input. Original rejection: {error}"
        )
        try:
            return operation(data.toarray())
        except Exception as retry_error:
            add_note = getattr(retry_error, "add_note", None)
            if callable(add_note):
                add_note(f"After one confirmed sparse-X retry during {context}: {error}")
            raise
