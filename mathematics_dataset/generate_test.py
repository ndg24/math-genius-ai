from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest as my_absltest
from mathematics_dataset import generate_math
from absl.testing import parameterized as my_parameterized
import six
from six.moves import range as six_range


class MathGenerateTest(my_parameterized.TestCase):

    def testCreateEntropyFunction(self):
        entropy_full = generate_math._create_entropy_function(0, 1)
        self.assertEqual(entropy_full((2, 3)), (2, 3))
        entropy_third = generate_math._create_entropy_function(2, 3)
        self.assertEqual(entropy_third((3, 6)), (5, 6))

    @my_parameterized.parameters('train', 'interpolate', 'extrapolate')
    def testMathGenerate(self, regime):
        generate_math.initialize_modules()
        for module in six.itervalues(generate_math.filtered_modules[regime]):
            for _ in six_range(3):
                question = module()
                str(question)


if __name__ == '__main__':
    my_absltest.main()