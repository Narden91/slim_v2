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
Correctness checks for slim_gsgp.classification (MS-SLIM margin loss and
adaptive inflate). These check the falsifiable mathematical claims in
MS_SLIM_formulation.md sections 2.2 and 11, not general test coverage.
"""

import torch

from slim_gsgp.classification.codes import encode_binary, decode_binary
from slim_gsgp.classification.losses import margin_loss, logistic_loss, code_regression_loss
from slim_gsgp.classification.adaptive_inflate import optimal_alpha
from slim_gsgp.classification.strategies import get_strategy, STRATEGIES


def test_encode_binary_maps_zero_one_to_pm1():
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    encoded = encode_binary(y)
    assert torch.equal(encoded, torch.tensor([-1.0, 1.0, 1.0, -1.0]))


def test_encode_binary_is_idempotent_on_pm1():
    y = torch.tensor([-1.0, 1.0, -1.0])
    assert torch.equal(encode_binary(y), y)


def test_decode_binary_sign_rule():
    y_pred = torch.tensor([-2.0, -0.001, 0.0, 0.001, 5.0])
    decoded = decode_binary(y_pred)
    assert torch.equal(decoded, torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))


def test_margin_loss_rejects_non_pm1_labels():
    loss = margin_loss(lam=0.1)
    y_true = torch.tensor([0.0, 1.0])
    y_pred = torch.tensor([0.5, -0.5])
    try:
        loss(y_true, y_pred)
        assert False, "expected ValueError for non-{-1,+1} labels"
    except ValueError:
        pass


def test_margin_loss_zero_at_lambda_zero_when_margins_satisfied():
    loss = margin_loss(lam=0.0)
    y_true = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    y_pred = torch.tensor([-2.0, 2.0, -1.0, 1.0])  # all margins exactly met or exceeded
    assert torch.isclose(loss(y_true, y_pred), torch.tensor(0.0), atol=1e-6)


def test_margin_loss_minimized_at_closed_form_optimum():
    """Formulation section 2.2: s* = y / (1 + lam), per-observation, for every lam > 0."""
    torch.manual_seed(0)
    for lam in (0.01, 0.1, 1.0, 5.0):
        loss = margin_loss(lam=lam)
        y_true = torch.tensor([1.0])
        s_star = y_true / (1.0 + lam)
        baseline = float(loss(y_true, s_star))
        for _ in range(50):
            perturbed = s_star + (torch.rand(1) - 0.5) * 2.0  # +/- 1.0 around optimum
            assert float(loss(y_true, perturbed)) >= baseline - 1e-6


def test_logistic_loss_stable_for_large_negative_margin():
    """Must not overflow to inf/nan where naive exp(-y*s) would."""
    loss = logistic_loss()
    y_true = torch.tensor([1.0])
    y_pred = torch.tensor([-1e8])
    result = loss(y_true, y_pred)
    assert torch.isfinite(result)


def test_code_regression_loss_matches_squared_error():
    loss = code_regression_loss()
    y_true = torch.tensor([-1.0, 1.0])
    y_pred = torch.tensor([0.0, 0.0])
    assert torch.isclose(loss(y_true, y_pred), torch.tensor(1.0))


def test_optimal_alpha_beats_random_alpha():
    """Formulation section 11: alpha* must not be worse than any other alpha."""
    torch.manual_seed(1)
    n, lam = 20, 0.1
    s = torch.randn(n)
    r = torch.randn(n)
    y = torch.tensor([1.0, -1.0]).repeat(n // 2)
    loss = margin_loss(lam=lam)

    alpha_star = optimal_alpha(s, r, y, lam)
    loss_star = float(loss(y, s + alpha_star * r))

    for _ in range(50):
        alpha = float((torch.rand(1) - 0.5) * 10.0)
        assert loss_star <= float(loss(y, s + alpha * r)) + 1e-5


def test_strategy_registry_has_matched_encode_decode():
    for name in STRATEGIES:
        strategy = get_strategy(name)
        assert strategy.fit_string in (
            "margin", "logistic", "code_regression", "sigmoid_rmse"
        )
        assert callable(strategy.encode)
        assert callable(strategy.decode)
