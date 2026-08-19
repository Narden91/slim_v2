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
Benchmark datasets for the MS-SLIM experimental campaign.

Chosen against the research plan's dataset design: balanced and imbalanced
binary problems, balanced and imbalanced multiclass problems, with deliberate
variation in sample size, dimensionality and class count. Iris and wine are
deliberately excluded from the campaign registry -- they are separable enough
that every method saturates, so they cannot discriminate between the losses
under comparison. They remain useful only as smoke tests.

Datasets are fetched from OpenML on first use and cached by scikit-learn under
``~/scikit_learn_data``. Fetching requires network access; ``load_dataset``
raises a clear error rather than silently substituting anything else.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

__all__ = ["DatasetSpec", "BINARY_DATASETS", "MULTICLASS_DATASETS", "DATASETS",
           "load_dataset", "describe_datasets"]


@dataclass(frozen=True)
class DatasetSpec:
    """
    A benchmark dataset and why it earns a place in the campaign.

    Attributes
    ----------
    name : str
        Registry key.
    loader : Callable[[], tuple]
        Returns ``(X, y)`` as float32 / int64 numpy arrays.
    task : str
        ``"binary"`` or ``"multiclass"``.
    n_classes : int
        Number of classes K.
    imbalance : str
        ``"balanced"``, ``"moderate"`` or ``"severe"``. Severe means the
        minority class is under roughly 5% of the data.
    rationale : str
        What this dataset contributes that the others do not.
    """
    name: str
    loader: Callable[[], tuple]
    task: str
    n_classes: int
    imbalance: str
    rationale: str
    tags: tuple = field(default_factory=tuple)


def _encode_targets(target):
    """Map arbitrary target labels to contiguous integer codes 0..K-1."""
    values = np.asarray(target).ravel()
    classes, codes = np.unique(values, return_inverse=True)
    return codes.astype("int64"), classes


def _openml(data_id: int):
    """Fetch an OpenML dataset by numeric id, as dense float32 / int64."""
    def _load():
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        frame_X = bunch.data
        # One-hot any categorical columns; SLIM terminals are numeric only.
        categorical = frame_X.select_dtypes(include=["category", "object"]).columns
        if len(categorical):
            import pandas as pd
            frame_X = pd.get_dummies(frame_X, columns=list(categorical), dummy_na=True)
        X = frame_X.astype("float32").to_numpy()
        # Median-impute; SLIM has no missing-value handling of its own.
        if np.isnan(X).any():
            medians = np.nanmedian(X, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            X = np.where(np.isnan(X), medians, X)
        codes, _ = _encode_targets(bunch.target)
        return X, codes

    return _load


def _sklearn(name: str):
    def _load():
        from sklearn import datasets as skd

        bunch = getattr(skd, name)()
        codes, _ = _encode_targets(bunch.target)
        return bunch.data.astype("float32"), codes

    return _load


BINARY_DATASETS = (
    DatasetSpec(
        name="breast_cancer", loader=_sklearn("load_breast_cancer"),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="Ships with the repo and is the incumbent classification benchmark "
                  "in this codebase, so it anchors the new results to the old ones. "
                  "Small (569 x 30) and comparatively easy: a floor, not a challenge.",
        tags=("small", "reference"),
    ),
    DatasetSpec(
        name="spambase", loader=_openml(44),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="4601 x 57, ~39% positive. Larger and genuinely harder than "
                  "breast_cancer, with many weakly informative features -- exercises "
                  "the search rather than the loss alone.",
        tags=("medium", "many-features"),
    ),
    DatasetSpec(
        name="phoneme", loader=_openml(1489),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="5404 x 5, ~29% positive. Low dimensionality but a strongly "
                  "nonlinear boundary, so performance depends on the evolved "
                  "structure rather than on feature count.",
        tags=("medium", "nonlinear"),
    ),
    DatasetSpec(
        name="credit_g", loader=_openml(31),
        task="binary", n_classes=2, imbalance="moderate",
        rationale="1000 rows, 20 raw features expanding to 74 after one-hot. "
                  "Noisy, low-signal and famously hard to beat the majority class on "
                  "-- distinguishes methods that overfit from methods that do not.",
        tags=("small", "categorical", "noisy"),
    ),
    DatasetSpec(
        name="pc4", loader=_openml(1049),
        task="binary", n_classes=2, imbalance="severe",
        rationale="1458 x 37 software-defect data, ~12% positive. Severe imbalance "
                  "where accuracy is uninformative and AUPRC/MCC carry the signal, "
                  "which is exactly what the balanced empirical risk targets.",
        tags=("small", "imbalanced"),
    ),
    DatasetSpec(
        name="mammography", loader=_openml(310),
        task="binary", n_classes=2, imbalance="severe",
        rationale="11183 x 6, ~2.3% positive. Extreme imbalance at scale; the "
                  "hardest test of whether the semantic regularizer and balanced "
                  "risk keep the minority class from being ignored.",
        tags=("large", "imbalanced"),
    ),
)

MULTICLASS_DATASETS = (
    DatasetSpec(
        name="waveform", loader=_openml(60),
        task="multiclass", n_classes=3, imbalance="balanced",
        rationale="5000 x 40, three balanced classes with heavy feature noise by "
                  "construction. K=3 keeps the semantic space 2-D while still "
                  "requiring a real decision boundary.",
        tags=("medium", "noisy"),
    ),
    DatasetSpec(
        name="vehicle", loader=_openml(54),
        task="multiclass", n_classes=4, imbalance="balanced",
        rationale="846 x 18, four classes, two of which overlap heavily. Small and "
                  "genuinely difficult -- a case where K-1 independent programs may "
                  "plausibly diverge from a shared structure.",
        tags=("small", "overlapping"),
    ),
    DatasetSpec(
        name="segment", loader=_openml(36),
        task="multiclass", n_classes=7, imbalance="balanced",
        rationale="2310 x 19, seven balanced classes. Raises K enough that the "
                  "cost gap between K-1 independent programs and one shared block "
                  "set becomes measurable, which is the point of Question 4.",
        tags=("medium", "high-K"),
    ),
    DatasetSpec(
        name="yeast", loader=_openml(181),
        task="multiclass", n_classes=10, imbalance="severe",
        rationale="1484 x 8, ten classes whose smallest holds 5 rows (0.34%). "
                  "The hardest case for a symmetric simplex, and "
                  "the one most likely to expose its limits -- which the plan "
                  "explicitly wants stated rather than hidden.",
        tags=("small", "high-K", "imbalanced"),
    ),
    DatasetSpec(
        name="pendigits", loader=_openml(32),
        task="multiclass", n_classes=10, imbalance="balanced",
        rationale="10992 x 16, ten balanced classes. The scale case: tests whether "
                  "either multiclass architecture remains tractable when both n and "
                  "K are large.",
        tags=("large", "high-K"),
    ),
)

DATASETS = {spec.name: spec for spec in BINARY_DATASETS + MULTICLASS_DATASETS}


def load_dataset(name: str):
    """
    Load a registered benchmark as ``(X, y)`` torch tensors.

    Parameters
    ----------
    name : str
        A key of ``DATASETS``.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``X`` float32 of shape (n, d); ``y`` float32 class codes in 0..K-1.
    """
    try:
        spec = DATASETS[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")

    try:
        X, y = spec.loader()
    except Exception as error:                      # network / OpenML failure
        raise RuntimeError(
            f"Could not load benchmark {name!r}: {error}. OpenML datasets need "
            "network access on first use; afterwards they are cached under "
            "~/scikit_learn_data."
        ) from error

    return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)


def describe_datasets(names=None) -> list:
    """
    Summarize registered benchmarks by actually loading them.

    Reports the real shape and class distribution rather than the values quoted
    in each ``rationale``, so the campaign's dataset table cannot drift away
    from the data it was written about.

    Parameters
    ----------
    names : iterable of str, optional
        Datasets to describe (default: all registered).

    Returns
    -------
    list of dict
        One record per dataset: name, task, n, d, K, class counts, minority share.
    """
    records = []
    for name in (names if names is not None else DATASETS):
        spec = DATASETS[name]
        X, y = load_dataset(name)
        counts = torch.bincount(y.long()).tolist()
        records.append({
            "dataset": name,
            "task": spec.task,
            "n": int(X.shape[0]),
            "d": int(X.shape[1]),
            "K": len(counts),
            "class_counts": counts,
            "minority_share": round(min(counts) / sum(counts), 4),
            "declared_imbalance": spec.imbalance,
        })
    return records
