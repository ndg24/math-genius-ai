from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl.testing import absltest
from mathematics_dataset.util import combinatorics

class CombinatoricsTest(absltest.TestCase):

  def testPositiveIntegersWithSum(self):
    result = combinatorics.uniform_positive_integers_with_sum(1, 1)
    self.assertEqual(result, [1])
    result = combinatorics.uniform_positive_integers_with_sum(2, 2)
    self.assertEqual(result, [1, 1])
    result = combinatorics.uniform_positive_integers_with_sum(1, 10)
    self.assertEqual(sum(result), 10)
    result = combinatorics.uniform_positive_integers_with_sum(2, 10)
    self.assertEqual(sum(result), 10)
    result = combinatorics.uniform_positive_integers_with_sum(0, 0)
    self.assertEqual(result, [])

  def testNonNegativeIntegersWithSum(self):
    result = combinatorics.uniform_non_negative_integers_with_sum(1, 0)
    self.assertEqual(result, [0])
    result = combinatorics.uniform_non_negative_integers_with_sum(2, 0)
    self.assertEqual(result, [0, 0])
    result = combinatorics.uniform_non_negative_integers_with_sum(3, 10)
    self.assertEqual(sum(result), 10)

  def testLogNumberBinaryTrees(self):
    self.assertAlmostEqual(
        combinatorics.log_number_binary_trees(0), math.log(1))
    self.assertAlmostEqual(
        combinatorics.log_number_binary_trees(1), math.log(1))
    self.assertAlmostEqual(
        combinatorics.log_number_binary_trees(2), math.log(2))
    self.assertAlmostEqual(
        combinatorics.log_number_binary_trees(3), math.log(5))
    self.assertAlmostEqual(
        combinatorics.log_number_binary_trees(4), math.log(14))

if __name__ == '__main__':
  absltest.main()
