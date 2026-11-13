"""Seeds and TensorFlow threading for reproducible neural-network fits.

The number of TensorFlow threads changes how sums are split across cores, and so
the rounding of every gradient step: with the same seed, the MDN stops at a
different training loss with 1, 2, 4 or the default number of threads (and an
explicit 4 differs from the default on a 4-core machine). The reference refit uses
TensorFlow's default thread pools on the 4-core GitHub Linux runner; it reproduces
two of the published training losses to six decimals. NIGERIA_RESILIENCE_TF_THREADS
pins both pools to another number (0, the default, keeps TensorFlow's choice).
"""
import os
import random
import warnings

import numpy as np
import tensorflow as tf

TF_THREADS = int(os.environ.get("NIGERIA_RESILIENCE_TF_THREADS", "0"))


def configure_tf_threads(n: int = TF_THREADS) -> None:
    """Pin TensorFlow's thread pools (must run before TensorFlow executes an op)."""
    if n <= 0:
        return
    try:
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(n)
    except RuntimeError:  # TensorFlow already initialised
        if tf.config.threading.get_intra_op_parallelism_threads() != n:
            warnings.warn("TensorFlow was initialised before its threads could be "
                          f"pinned to {n}; the refit may depend on the machine.")


configure_tf_threads()


def setup_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if os.environ.get("NIGERIA_RESILIENCE_OP_DETERMINISM", "1") == "1":
        try:  # in addition to TF_DETERMINISTIC_OPS=1, as in the PNAS package
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass
