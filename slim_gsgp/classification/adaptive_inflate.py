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
Adaptive inflate mutation for margin_loss (MS_SLIM_formulation.md, section 11).

Instead of drawing the new block's mutation step ``ms`` at random, solve for
the step that minimizes ``margin_loss(s + alpha * r)`` directly, where ``s``
is the parent's current (collapsed) semantics and ``r`` is the candidate
block's raw semantics. This is a 1-D strictly-convex piecewise-quadratic
problem; solved here by a few fixed-point iterations on the active hinge set
rather than an exact breakpoint sweep (see integration plan, section 3).

Only defined for additive ("sum" operator) SLIM: the block's contribution to
the individual's semantics is ``alpha * r`` -- linear in ``alpha`` -- only
under the sum operator. Under "prod" the update is multiplicative
(``1 + alpha * r``), which is out of scope for this method (formulation
section 12).
"""

import torch

__all__ = ["optimal_alpha", "adaptive_inflate"]


def optimal_alpha(
    s: torch.Tensor,
    r: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    iters: int = 3,
    balanced: bool = False,
) -> float:
    """
    Solve ``argmin_alpha margin_loss(s + alpha * r)`` for margin_loss's
    ``[1 - y*(s + alpha*r)]_+^2 + lam*(s + alpha*r)^2``.

    Fixed-point iteration on the active hinge set (formulation section 11):
    on a fixed active set the objective is an exact quadratic in ``alpha``,
    solved in closed form; recomputing the active set after each solve
    converges in a handful of steps since the set only shrinks or grows at
    isolated breakpoints.

    Parameters
    ----------
    s : torch.Tensor
        Parent individual's current train semantics (1-D, length n).
    r : torch.Tensor
        Candidate block's raw train semantics (1-D, length n).
    y : torch.Tensor
        Labels in ``{-1, +1}`` (1-D, length n).
    lam : float
        Semantic regularization strength, matching the ``margin_loss`` in use.
    iters : int, optional
        Number of fixed-point iterations (default 3).

    Returns
    -------
    float
        The solved mutation step ``alpha*``.
    """
    alpha = 0.0
    if balanced:
        pos, neg = y > 0, y < 0
        w = torch.empty_like(y)
        w[pos] = 0.5 / pos.sum().clamp(min=1)
        w[neg] = 0.5 / neg.sum().clamp(min=1)
    else:
        w = torch.ones_like(y) / len(y)

    for _ in range(iters):
        active = (1.0 - y * (s + alpha * r)) > 0
        yr = y * r
        num = torch.sum(w * active * yr * (1.0 - y * s)) - lam * torch.sum(w * s * r)
        den = torch.sum(w * active * r * r) + lam * torch.sum(w * r * r)
        alpha = float(num / den) if den > 0 else 0.0
    return alpha


def adaptive_inflate(base_inflate, y_train: torch.Tensor, lam: float, operator: str = "sum", balanced: bool = False):
    """
    Wrap a SLIM ``inflate_mutator`` to use ``optimal_alpha`` instead of a
    random mutation step, for the ``margin_loss`` fitness.

    Calls ``base_inflate`` once with ``ms=1`` to obtain the candidate block's
    raw semantics ``r`` (the block delta at unit step, since the sum-operator
    delta rules are linear in ``ms``), solves for ``alpha*``, then rescales
    the resulting offspring's new block and total semantics by ``alpha*`` --
    avoiding a second, more expensive call into ``base_inflate``.

    Parameters
    ----------
    base_inflate : Callable
        An inflate-mutation function as built by
        ``slim_gsgp.algorithms.SLIM_GSGP.operators.mutators.inflate_mutation``.
    y_train : torch.Tensor
        Training labels in ``{-1, +1}``, matching the individuals being mutated.
    lam : float
        Semantic regularization strength, matching the ``margin_loss`` in use.
    operator : str, optional
        Must be "sum" -- adaptive inflate is only defined for additive SLIM
        (default "sum").

    Returns
    -------
    Callable
        A drop-in replacement inflate mutator with the same signature as
        ``base_inflate``.
    """
    if operator != "sum":
        raise ValueError("adaptive_inflate only supports the 'sum' operator (additive SLIM)")

    def _adaptive_inflate(individual, ms, X, **kwargs):
        if hasattr(individual, "get_train_semantics_collapsed"):
            s = individual.get_train_semantics_collapsed(torch.sum, dim=0)
        else:
            s = torch.sum(individual.train_semantics, dim=0)
        offs = base_inflate(individual, 1.0, X, **kwargs)
        r = offs.train_semantics[-1]  # block delta at unit step (linear in ms under "sum")

        alpha = optimal_alpha(s, r, y_train, lam, balanced=balanced)

        offs.train_semantics[-1] = r * alpha
        if offs.test_semantics is not None:
            offs.test_semantics[-1] = offs.test_semantics[-1] * alpha
        if hasattr(offs, "collection"):
            offs.collection[-1].structure[-1] = alpha
        return offs

    return _adaptive_inflate
