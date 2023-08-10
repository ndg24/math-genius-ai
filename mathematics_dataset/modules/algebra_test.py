from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from absl.testing import absltest
from mathematics_dataset.modules import algebra_math
from mathematics_dataset.sample import polynomials_math
from six.moves import range as six_range
import sympy


class AlgebraTest(absltest.TestCase):

  def testPolynomialCoeffsWithRoots(self):
    coeffs = algebra_math._polynomial_coeffs_with_roots([1, 2], scale_entropy=0.0)
    self.assertEqual(coeffs, [2, -3, 1])

  def testPolynomialRoots(self):
    variable = sympy.Symbol('x')
    for _ in six_range(10):
      roots = random.sample(list(six_range(-9, 10)), 3)
      coeffs = algebra_math._polynomial_coeffs_with_roots(roots, scale_entropy=10.0)
      polynomial = polynomials_math.coefficients_to_polynomial(coeffs, variable)
      calc_roots = sympy.polys.polytools.real_roots(polynomial)
      self.assertEqual(calc_roots, sorted(roots))


if __name__ == '__main__':
  absltest.main()
