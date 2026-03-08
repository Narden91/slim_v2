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
Binary classification utilities for SLIM_GSGP.

This subpackage follows the approach of Bakurov et al. (2022) for adapting
GSGP-based algorithms to binary classification:

- During training: the tree outputs are passed through a sigmoid to bound them
  in [0, 1], and RMSE is computed against binary labels.
- At prediction time: negative outputs are mapped to class 0 and non-negative
  outputs to class 1 via ``binary_sign_transform``.

Relevant fitness function:
    ``sigmoid_rmse``  – registered in ``slim_gsgp.config.slim_config.fitness_function_options``

Relevant prediction helper:
    ``binary_sign_transform``  – imported from ``slim_gsgp.evaluators.fitness_functions``
"""

from slim_gsgp.evaluators.fitness_functions import sigmoid_rmse, binary_sign_transform

__all__ = ["sigmoid_rmse", "binary_sign_transform"]
