# Copyright 2018 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mathematics_dataset.modules.calculus."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import mathematics_dataset.modules.calculus as calculus

class CalculusTest(tf.test.TestCase):

  def testSampleIntegrand(self):
    coefficients = [[0, 1], [2, 0], [3, 0]]
    derivative_order = 1
    derivative_axis = 0
    expected = [[0, 1], [1, 0], [1, 0]]
    entropy = 4
    result = calculus._sample_integrand(
        coefficients, derivative_order, derivative_axis, entropy)
    result = result[1:, :]
    self.assertAllEqual(result, expected)

if __name__ == '__main__':
  tf.test.main()
