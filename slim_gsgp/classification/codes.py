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
Label encoding/decoding for margin-based classification (MS-SLIM).

Binary classification is the K=2 case of the simplex-code formulation
(MS_SLIM_formulation.md, section 7), so its codes are hardcoded here as
``{-1, +1}`` rather than generated from a general K-class simplex routine
that this repo does not otherwise need yet (see integration plan section 0,
anti-duplication rule 4).
"""

import torch

_POS, _NEG = 1.0, -1.0


def encode_binary(y: torch.Tensor) -> torch.Tensor:
    """
    Encode arbitrary two-valued labels as ``{-1, +1}``.

    The larger of the two distinct values in ``y`` is mapped to ``+1`` and
    the smaller to ``-1`` (so ``{0, 1}`` labels map the usual way, with
    ``1 -> +1``). Already-encoded ``{-1, +1}`` input is returned unchanged.

    Parameters
    ----------
    y : torch.Tensor
        Labels with exactly two distinct values.

    Returns
    -------
    torch.Tensor
        Float tensor with values in ``{-1.0, +1.0}``.
    """
    values = torch.unique(y)
    if values.numel() != 2:
        raise ValueError(f"encode_binary expects exactly 2 distinct labels, got {values.tolist()}")
    hi = values.max()
    return torch.where(y == hi, torch.tensor(_POS), torch.tensor(_NEG)).float()


def decode_binary(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Decode raw MS-SLIM semantics to ``{-1, +1}`` class predictions.

    Prediction rule from MS_SLIM_formulation.md, section 2: ``sign(P(x))``,
    with the convention that ``P(x) == 0`` is assigned to the positive class.

    Parameters
    ----------
    y_pred : torch.Tensor
        Raw semantics from a trained individual.

    Returns
    -------
    torch.Tensor
        Tensor of ``{-1.0, +1.0}``.
    """
    return torch.where(y_pred >= 0, torch.tensor(_POS), torch.tensor(_NEG)).float()
