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
Fitness functions for margin-based classification (MS-SLIM) and the raw-score
baselines needed to isolate its effect (MS_SLIM_formulation.md, section 14).

All of these operate directly on raw SLIM semantics -- no sigmoid or softmax
is inserted -- and follow the same factory convention as
``slim_gsgp.evaluators.fitness_functions.sigmoid_rmse``: a hyperparameter
constructor returns a ``(y_true, y_pred) -> torch.Tensor`` callable with a
descriptive ``__name__`` for logging.
"""

import torch
import torch.nn.functional as F

_BINARY_LABELS = {-1.0, 1.0}


def _check_binary_labels(y_true: torch.Tensor) -> None:
    labels = set(torch.unique(y_true).tolist())
    if not labels <= _BINARY_LABELS:
        raise ValueError(
            f"expected labels in {{-1, +1}}, got {sorted(labels)}; "
            "use slim_gsgp.classification.codes.encode_binary(y) first"
        )


def _class_balanced_mean(per_obs: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Average per-class means instead of the per-observation mean (formulation section 8)."""
    pos, neg = y_true > 0, y_true < 0
    return 0.5 * (per_obs[pos].mean() + per_obs[neg].mean())


def margin_loss(lam: float = 0.01, balanced: bool = False):
    """
    Squared-hinge margin loss with semantic L2 regularization.

    ``L(s) = mean( [1 - y*s]_+^2 + lam * s^2 )``, ``y in {-1, +1}``.
    Strictly convex in ``s`` with unique optimum ``s* = y / (1 + lam)``
    (MS_SLIM_formulation.md, section 2).

    Parameters
    ----------
    lam : float, optional
        Semantic regularization strength (default 0.01). Must be positive
        for the optimum to be unique and bounded.
    balanced : bool, optional
        If True, average per-class losses instead of over all observations,
        for class-imbalanced data (formulation section 8).

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    """
    def _margin_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        _check_binary_labels(y_true)
        violation = torch.clamp(1.0 - y_true * y_pred, min=0.0) ** 2
        per_obs = violation + lam * y_pred ** 2
        return _class_balanced_mean(per_obs, y_true) if balanced else per_obs.mean()

    _margin_loss.__name__ = f"margin_loss(lam={lam}, balanced={balanced})"
    return _margin_loss


def logistic_loss():
    """
    Raw-score logistic loss baseline: ``mean(log(1 + exp(-y*s)))``, ``y in {-1, +1}``.

    Computed via ``softplus`` for numerical stability -- the naive ``exp``
    form overflows once ``y*s`` is strongly negative, which happens routinely
    since SLIM semantics are only clamped at +/-1e12.

    Required baseline to distinguish "removing the sigmoid helps" from
    "the margin loss specifically helps" (formulation section 14, Question 1).

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    """
    def _logistic_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        _check_binary_labels(y_true)
        return F.softplus(-y_true * y_pred).mean()

    _logistic_loss.__name__ = "logistic_loss"
    return _logistic_loss


def code_regression_loss():
    """
    Plain regression onto the class code: ``mean((y - s)^2)``.

    As a binary baseline, ``y in {-1, +1}``: isolates the effect of the
    one-sided hinge margin from the effect of using class codes and a
    distance-based prediction rule at all (formulation section 14,
    Question 2). Also reused, unmodified, as the per-coordinate loss for
    independent-coordinate multiclass MS-SLIM (``classification.multiclass``,
    integration plan phase M1), where ``y`` is a continuous simplex-code
    coordinate rather than +-1 -- squared error is well-defined either way,
    so no separate label check is applied here.

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    """
    def _code_regression_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_true - y_pred) ** 2)

    _code_regression_loss.__name__ = "code_regression_loss"
    return _code_regression_loss
