# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Classification utilities for SLIM_GSGP.

Two families of technique are available, chosen via
``slim_gsgp.main_slim.slim(fitness_function=...)``:

- **Sigmoid + RMSE** (Bakurov et al. 2022): squashes raw outputs through a
  sigmoid before comparing to {0,1} labels. Kept for backward compatibility;
  see ``slim_gsgp.evaluators.fitness_functions.sigmoid_rmse``.
- **MS-SLIM / margin loss** (this module): evaluates classification loss
  directly on raw semantics, with a squared-hinge margin term. See
  ``MS_SLIM_formulation.md`` for the full derivation.

The recommended entry point is ``get_strategy(name)``, which returns a
``ClassificationStrategy`` binding the label encoding, fitness function, and
prediction decoding that belong together for a given technique -- so you
cannot accidentally decode margin-loss semantics with a sigmoid threshold,
or vice versa.

Example
-------
>>> from slim_gsgp.classification import get_strategy
>>> from slim_gsgp.main_slim import slim
>>> strategy = get_strategy("margin")
>>> y_train_encoded = strategy.encode(y_train)
>>> model = slim(X_train=X_train, y_train=y_train_encoded,
...               fitness_function=strategy.fit_string, lam=0.01)
>>> y_pred = strategy.decode(model.predict(X_test))
"""

from slim_gsgp.classification.codes import encode_binary, decode_binary
from slim_gsgp.classification.losses import margin_loss, logistic_loss, code_regression_loss
from slim_gsgp.classification.strategies import ClassificationStrategy, STRATEGIES, get_strategy
from slim_gsgp.classification.adaptive_inflate import optimal_alpha, adaptive_inflate
from slim_gsgp.classification.experiments import run_experiment
from slim_gsgp.classification.multiclass import (
    simplex_codes, MulticlassResult, fit_multiclass, predict_multiclass,
)
from slim_gsgp.evaluators.fitness_functions import sigmoid_rmse, binary_sign_transform

__all__ = [
    "encode_binary",
    "decode_binary",
    "margin_loss",
    "logistic_loss",
    "code_regression_loss",
    "ClassificationStrategy",
    "STRATEGIES",
    "get_strategy",
    "optimal_alpha",
    "adaptive_inflate",
    "run_experiment",
    "simplex_codes",
    "MulticlassResult",
    "fit_multiclass",
    "predict_multiclass",
    "sigmoid_rmse",
    "binary_sign_transform",
]
