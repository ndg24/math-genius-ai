from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from absl.testing import absltest
from absl.testing import parameterized
from mathematics_dataset.sample import arithmetic_expr_generator as arithmetic
from mathematics_dataset.sample import num_sample
from mathematics_dataset.sample import math_ops
from six.moves import range
import sympy

class ArithmeticTest(parameterized.TestCase):

  def testArithmetic(self):
    for _ in range(1000):
      target = num_sample.integer_or_rational(4, signed=True)
      entropy = 8.0
      expr = arithmetic.generate_expression(target, entropy)
      self.assertEqual(sympy.sympify(expr), target)

  def testArithmeticLength(self):
    for _ in range(1000):
      target = num_sample.integer_or_rational(4, signed=True)
      entropy = 8.0
      length = random.randint(2, 10)
      expr = arithmetic.generate_expression(target, entropy, length)
      actual_length = len(math_ops.constants_in_expression(expr)) - 1
      self.assertEqual(actual_length, length)

if __name__ == '__main__':
  absltest.main()
