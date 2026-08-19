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
adaptive inflate). These check the mathematical claims in
MS_SLIM_formulation.md sections 2.2 and 11, not general test coverage.
"""

import torch

from slim_gsgp.classification.codes import encode_binary, encode_zero_one, decode_binary
from slim_gsgp.classification.losses import (
    margin_loss, logistic_loss, code_regression_loss, multiclass_margin_loss,
)
from slim_gsgp.classification.adaptive_inflate import optimal_alpha
from slim_gsgp.classification.strategies import get_strategy, STRATEGIES
from slim_gsgp.classification.multiclass import simplex_codes


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


def test_balanced_margin_loss_never_returns_nan_on_single_class_batch():
    """A NaN fitness is silently chosen as best by np.argmin in get_best_min."""
    loss = margin_loss(lam=0.1, balanced=True)
    y_true = torch.tensor([1.0, 1.0, 1.0])  # negative class absent
    y_pred = torch.tensor([0.5, 0.5, 0.5])
    assert torch.isfinite(loss(y_true, y_pred))


def test_encode_zero_one_maps_pm1_to_zero_one():
    """sigmoid_rmse trains on {0,1}; {-1,+1} input must be converted, not passed through."""
    assert torch.equal(encode_zero_one(torch.tensor([-1.0, 1.0, -1.0])),
                       torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(encode_zero_one(torch.tensor([0.0, 1.0])),
                       torch.tensor([0.0, 1.0]))


def test_sigmoid_rmse_strategy_encodes_to_zero_one():
    encoded = get_strategy("sigmoid_rmse").encode(torch.tensor([-1.0, 1.0]))
    assert set(encoded.tolist()) <= {0.0, 1.0}


def test_simplex_codes_is_regular_simplex():
    """Formulation section 3: unit norm, sum to zero, equal pairwise inner product."""
    for k in (2, 3, 4, 5, 8):
        codes = simplex_codes(k)
        assert codes.shape == (k, k - 1)
        assert torch.allclose(codes.norm(dim=1), torch.ones(k), atol=1e-5)
        assert torch.allclose(codes.sum(dim=0), torch.zeros(k - 1), atol=1e-5)

        gram = codes @ codes.T
        off_diag = gram[~torch.eye(k, dtype=torch.bool)]
        expected = -1.0 / (k - 1)
        assert torch.allclose(off_diag, torch.full_like(off_diag, expected), atol=1e-5)


def test_simplex_codes_k2_matches_binary_codes():
    """K=2 must reduce to the {-1,+1} codes margin_loss uses (formulation section 7)."""
    codes = simplex_codes(2)
    assert torch.equal(codes, torch.tensor([[-1.0], [1.0]]))


def test_main_slim_imports_without_priming_classification_package():
    """
    slim_config imports classification.losses, and classification's heavy modules
    import main_slim -- an eager re-export makes that a cycle that only stays
    hidden when something imports slim_gsgp.classification first. Import
    main_slim in a clean interpreter to catch a regression.
    """
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c", "from slim_gsgp.main_slim import slim"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_lazy_exports_still_reachable_from_package():
    import slim_gsgp.classification as classification
    for name in ("run_experiment", "fit_multiclass", "predict_multiclass", "MulticlassResult"):
        assert hasattr(classification, name)


def test_multiclass_margin_loss_optimum_at_scaled_class_code():
    """Formulation section 6.2: s_i* = c_{y_i} / (1 + lam), independent of K."""
    for k in (2, 3, 4, 5):
        codes = simplex_codes(k)
        y_rows = torch.arange(k)
        lam = 0.1
        loss = multiclass_margin_loss(codes, y_rows, lam=lam)
        optimum = codes[y_rows] / (1.0 + lam)
        baseline = float(loss(optimum))
        torch.manual_seed(0)
        for _ in range(100):
            perturbed = optimum + torch.randn(k, k - 1) * 0.3
            assert float(loss(perturbed)) >= baseline - 1e-6


def test_multiclass_margin_loss_reduces_to_binary_case():
    """Formulation section 7: K=2 must reproduce the binary margin loss exactly."""
    codes = simplex_codes(2)
    y_rows = torch.tensor([0, 1, 1, 0])
    y_binary = codes[y_rows].squeeze(1)          # {-1,+1}
    lam = 0.2
    joint = multiclass_margin_loss(codes, y_rows, lam=lam)
    binary = margin_loss(lam=lam)

    torch.manual_seed(3)
    for _ in range(20):
        s = torch.randn(4)
        assert torch.isclose(joint(s.unsqueeze(1)), binary(y_binary, s), atol=1e-6)


def test_fit_coefficients_reaches_convex_optimum():
    """A is convex given fixed block semantics; LBFGS must not improve on it."""
    from slim_gsgp.classification.shared_blocks import fit_coefficients

    torch.manual_seed(0)
    n_classes, n, n_blocks = 3, 80, 4
    codes = simplex_codes(n_classes)
    y_rows = torch.randint(0, n_classes, (n,))
    loss = multiclass_margin_loss(codes, y_rows, lam=0.05)
    R = torch.randn(n_blocks, n)

    A = fit_coefficients(R, loss, n_classes, iters=200)
    fitted = float(loss(R.T @ A))

    refined = A.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([refined], max_iter=200)

    def closure():
        optimizer.zero_grad()
        value = loss(R.T @ refined)
        value.backward()
        return value

    optimizer.step(closure)
    assert fitted <= float(loss(R.T @ refined.detach())) + 1e-4


def test_fit_shared_blocks_rejects_multiplicative_slim():
    from slim_gsgp.classification.shared_blocks import fit_shared_blocks
    X = torch.randn(20, 3)
    y = torch.tensor([0.0] * 7 + [1.0] * 7 + [2.0] * 6)
    try:
        fit_shared_blocks(X, y, slim_version="SLIM*ABS", pop_size=4, n_iter=1)
        assert False, "expected ValueError for multiplicative SLIM"
    except ValueError:
        pass


def test_fit_shared_blocks_learns_a_separable_problem():
    """End-to-end: shared blocks + joint loss must beat chance on clean data."""
    from slim_gsgp.classification.shared_blocks import fit_shared_blocks

    torch.manual_seed(0)
    centers = torch.tensor([[3.0, 0.0], [-3.0, 3.0], [-3.0, -3.0]])
    y_rows = torch.arange(3).repeat_interleave(25)
    X = centers[y_rows] + torch.randn(75, 2) * 0.3
    y = y_rows.float()

    result = fit_shared_blocks(X, y, pop_size=20, n_iter=8, lam=0.01, seed=0,
                               coefficient_iters=40)
    accuracy = float((result.predict(X) == y).float().mean())
    assert accuracy > 0.6, f"well-separated 3-class problem only reached {accuracy}"
    assert result.coefficients.shape == (result.individual.size, 2)
