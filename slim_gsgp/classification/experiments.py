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
Experiment harness for comparing classification strategies on SLIM_GSGP.

Runs the same dataset/splits across multiple seeds and strategies, so results
are paired and comparable (integration plan, section 5). Metrics come from
``sklearn.metrics`` -- no metric math is reimplemented here.
"""

import time

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)

from slim_gsgp.main_slim import slim
from slim_gsgp.utils.utils import train_test_split
from slim_gsgp.classification.strategies import get_strategy

__all__ = ["run_experiment"]


def _semantic_diagnostics(model, X_train, y_train_enc) -> dict:
    """MS-SLIM diagnostics from the research plan: semantic norm and margin stats."""
    with torch.no_grad():
        s = model.predict(X_train)
    margins = y_train_enc * s
    return {
        "semantic_norm": float(torch.linalg.norm(s)),
        "mean_margin": float(margins.mean()),
        "margin_ge_1_frac": float((margins >= 1).float().mean()),
    }


def run_experiment(
    X,
    y,
    dataset_name: str,
    strategy_name: str,
    seeds=range(30),
    p_test: float = 0.2,
    p_val: float = 0.25,
    **slim_kwargs,
) -> pd.DataFrame:
    """
    Train and evaluate one classification strategy on one dataset across seeds.

    For each seed: split into train/val/test with that seed (identical splits
    are reproduced across strategies run with the same seed, since the split
    is re-derived from the seed rather than cached -- integration plan,
    section 5, point 1), train with the validation set passed as the model's
    internal ``X_test``/``y_test`` (never the held-out test set -- point 2),
    then score on the untouched test set.

    Parameters
    ----------
    X, y : array-like
        Full dataset, not yet split.
    dataset_name : str
        Passed through to ``slim()`` for logging.
    strategy_name : str
        A key in ``slim_gsgp.classification.strategies.STRATEGIES``.
    seeds : iterable of int, optional
        Seeds to run (default ``range(30)``, per the research plan's
        minimum-30-runs protocol).
    p_test : float, optional
        Fraction held out as the final test set (default 0.2).
    p_val : float, optional
        Fraction of the remaining train data held out as validation,
        passed to ``slim()`` as ``X_test``/``y_test`` (default 0.25).
    **slim_kwargs
        Forwarded to ``slim_gsgp.main_slim.slim`` (e.g. ``pop_size``,
        ``n_iter``, ``lam``, ``slim_version``, ``use_adaptive_inflate``).

    Returns
    -------
    pandas.DataFrame
        One row per seed: accuracy, balanced_accuracy, f1, mcc, auroc, auprc,
        nodes_count, n_blocks, train_time_s, plus semantic diagnostics for
        margin-based strategies.
    """
    X = torch.as_tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
    y = torch.as_tensor(y, dtype=torch.float32) if not torch.is_tensor(y) else y
    strategy = get_strategy(strategy_name)

    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=p_test, seed=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, p_test=p_val, seed=seed)

        y_train_enc = strategy.encode(y_train)
        y_val_enc = strategy.encode(y_val)

        start = time.time()
        model = slim(
            X_train=X_train, y_train=y_train_enc,
            X_test=X_val, y_test=y_val_enc,
            dataset_name=dataset_name,
            fitness_function=strategy.fit_string,
            seed=seed,
            **slim_kwargs,
        )
        train_time_s = time.time() - start

        raw_scores = model.predict(X_test)
        y_pred = strategy.decode(raw_scores).numpy()
        y_true = strategy.encode(y_test).numpy()

        # AUROC/AUPRC rank observations, so they take the raw semantics rather
        # than decoded labels -- thresholding first would throw away exactly the
        # ordering they measure. Required for imbalanced data by the research plan.
        positive = (y_true == y_true.max())
        row = {
            "dataset": dataset_name,
            "strategy": strategy_name,
            "seed": seed,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, pos_label=1.0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "auroc": roc_auc_score(positive, raw_scores.numpy()),
            "auprc": average_precision_score(positive, raw_scores.numpy()),
            "nodes_count": model.nodes_count,
            "n_blocks": model.size,
            "train_time_s": train_time_s,
        }
        if strategy.fit_string == "margin":
            row.update(_semantic_diagnostics(model, X_train, y_train_enc))
        rows.append(row)

    return pd.DataFrame(rows)
