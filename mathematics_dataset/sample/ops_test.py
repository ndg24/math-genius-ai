from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from mathematics_dataset.sample import custom_ops
from six.moves import range
import sympy

class CustomOpsTest(absltest.TestCase):

  def testNegOp(self):
    op = custom_ops.NegOp(2)
    self.assertEqual(str(op), '-2')
    self.assertEqual(op.sympy(), -2)

    op = custom_ops.AddOp(custom_ops.NegOp(2), 3)
    self.assertEqual(str(op), '-2 + 3')
    self.assertEqual(op.sympy(), 1)

    op = custom_ops.AddOp(3, custom_ops.NegOp(2))
    self.assertEqual(str(op), '3 - 2')
    self.assertEqual(op.sympy(), 1)

    op = custom_ops.AddOp(custom_ops.AddOp(custom_ops.NegOp(2), 5), 3)
    self.assertEqual(str(op), '-2 + 5 + 3')
    self.assertEqual(op.sympy(), 6)

    op = custom_ops.AddOp(3, custom_ops.AddOp(custom_ops.IdentityOp(custom_ops.NegOp(2)), 5))
    self.assertEqual(str(op), '3 - 2 + 5')
    self.assertEqual(op.sympy(), 6)

    op = custom_ops.AddOp(3, custom_ops.AddOp(2, custom_ops.NegOp(5)))
    self.assertEqual(str(op), '3 + 2 - 5')
    self.assertEqual(op.sympy(), 0)

  def testAddOp(self):
    add = custom_ops.AddOp()
    self.assertEqual(str(add), '0')
    self.assertEqual(add.sympy(), 0)

    add = custom_ops.AddOp(2, 3)
    self.assertEqual(str(add), '2 + 3')
    self.assertEqual(add.sympy(), 5)

    add = custom_ops.AddOp(custom_ops.AddOp(1, 2), 3)
    self.assertEqual(str(add), '1 + 2 + 3')
    self.assertEqual(add.sympy(), 6)

  def testSubOp(self):
    sub = custom_ops.SubOp(2, 3)
    self.assertEqual(str(sub), '2 - 3')
    self.assertEqual(sub.sympy(), -1)

    sub = custom_ops.SubOp(custom_ops.SubOp(1, 2), 3)
    self.assertEqual(str(sub), '1 - 2 - 3')
    self.assertEqual(sub.sympy(), -4)

    sub = custom_ops.SubOp(1, custom_ops.SubOp(2, 3))
    self.assertEqual(str(sub), '1 - (2 - 3)')
    self.assertEqual(sub.sympy(), 2)

    sub = custom_ops.SubOp(custom_ops.NegOp(1), 2)
    self.assertEqual(str(sub), '-1 - 2')
    self.assertEqual(sub.sympy(), -3)

  def testMulOp(self):
    mul = custom_ops.MulOp()
    self.assertEqual(str(mul), '1')
    self.assertEqual(mul.sympy(), 1)

    mul = custom_ops.MulOp(2, 3)
    self.assertEqual(str(mul), '2*3')
    self.assertEqual(mul.sympy(), 6)

    mul = custom_ops.MulOp(custom_ops.IdentityOp(custom_ops.ConstantOp(-2)), 3)
    self.assertEqual(str(mul), '-2*3')
    self.assertEqual(mul.sympy(), -6)

    mul = custom_ops.MulOp(custom_ops.AddOp(1, 2), 3)
    self.assertEqual(str(mul), '(1 + 2)*3')
    self.assertEqual(mul.sympy(), 9)

    mul = custom_ops.MulOp(custom_ops.MulOp(2, 3), 5)
    self.assertEqual(str(mul), '2*3*5')
    self.assertEqual(mul.sympy(), 30)

  def testDivOp(self):
    div = custom_ops.DivOp(2, 3)
    self.assertEqual(str(div), '2/3')
    self.assertEqual(div.sympy(), sympy.Rational(2, 3))

    div = custom_ops.DivOp(2, sympy.Rational(4, 5))
    self.assertEqual(str(div), '2/(4/5)')
    self.assertEqual(div.sympy(), sympy.Rational(5, 2))

    div = custom_ops.DivOp(1, custom_ops.DivOp(2, 3))
    self.assertEqual(str(div), '1/(2/3)')
    self.assertEqual(div.sympy(), sympy.Rational(3, 2))

    div = custom_ops.DivOp(custom_ops.DivOp(2, 3), 4)
    self.assertEqual(str(div), '(2/3)/4')
    self.assertEqual(div.sympy(), sympy.Rational(1, 6))

    div = custom_ops.DivOp(2, custom_ops.MulOp(3, 4))
    self.assertEqual(str(div), '2/(3*4)')

    div = custom_ops.DivOp(2, sympy.Function('f')(sympy.Symbol('x')))
    self.assertEqual(str(div), '2/f(x)')

  def testPowOp(self):
    pow_ = custom_ops.PowOp(2, 3)
    self.assertEqual(str(pow_), '2**3')
    self.assertEqual(pow_.sympy(), 8)

    pow_ = custom_ops.PowOp(4, sympy.Rational(1, 2))
    self.assertEqual(str(pow_), '4**(1/2)')
    self.assertEqual(pow_.sympy(), 2)

    pow_ = custom_ops.PowOp(sympy.Rational(1, 2), 3)
    self.assertEqual(str(pow_), '(1/2)**3')
    self.assertEqual(pow_.sympy(), 1/8)

    pow_ = custom_ops.PowOp(3, custom_ops.PowOp(2, 1))
    self.assertEqual(str(pow_), '3**(2**1)')
    self.assertEqual(pow_.sympy(), 9)

    pow_ = custom_ops.PowOp(custom_ops.PowOp(2, 3), 4)
    self.assertEqual(str(pow_), '(2**3)**4')
    self.assertEqual(pow_.sympy(), 4096)

    pow_ = custom_ops.PowOp(-5, 2)
    self.assertEqual(str(pow_), '(-5)**2')
    self.assertEqual(pow_.sympy(), 25)

  def testEqOp(self):
    op = custom_ops.EqOp(custom_ops.AddOp(2, 3), 4)
    self.assertEqual(str(op), '2 + 3 = 4')
    self.assertEqual(op.sympy(), False)

  def testDescendants(self):
    constants = [custom_ops.ConstantOp(i) for i in range(6)]

    expression = custom_ops.SubOp(
        custom_ops.DivOp(
            custom_ops.AddOp(
                constants[0],
                custom_ops.MulOp(
                    constants[1],
                    custom_ops.PowOp(
                        constants[2],
                        constants[3]))),
            constants[4]),
        constants[5])
    descendants = expression.descendants()
    descendants = custom_ops._flatten(descendants)

    for constant in constants:
      self.assertIn(constant, descendants)
      self.assertEqual(descendants.count(constant), 1)

    self.assertEqual(constants[0].descendants(), [constants[0]])

    constant = custom_ops.ConstantOp(3)
    expression = custom_ops.NegOp(constant)
    self.assertEqual(set(expression.descendants()), set([constant, expression]))

  def testNumConstants(self):
    constant = custom_ops.ConstantOp(3)
    expression = custom_ops.NegOp(constant)
    constants = custom_ops.num_constants([expression])
    self.assertEqual(constants, [constant])

if __name__ == '__main__':
  absltest.main()
