from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random

from mathematics_dataset.sample import number as num_sample
from mathematics_dataset.sample import ops as math_ops
from mathematics_dataset.util import combinatorics_util
import numpy as np
import six
from six.moves import zip
import sympy

class ExpressionArgs(collections.namedtuple('ExpressionArgs', ('expr_count', 'expr_entropy'))):

  def decompose(self, fraction=1):
    entropy_fraction = fraction * self.expr_entropy / self.expr_count
    new_expression_args = ExpressionArgs(self.expr_count, self.expr_entropy - entropy_fraction)
    return entropy_fraction, new_expression_args

  def split(self, args):
    non_integer_count = sum(not arg.is_Integer for arg in args)
    assert non_integer_count <= self.expr_count - 1
    count_split = combinatorics_util.uniform_non_negative_integers_with_sum(len(args), (self.expr_count - 1) - non_integer_count)
    for i, arg in enumerate(args):
      if not arg.is_Integer:
        count_split[i] += 1
    if all(count == 0 for count in count_split):
      assert self.expr_entropy == 0
      entropies = np.zeros(len(count_split))
    else:
      entropies = (
          np.random.dirichlet(np.maximum(1e-9, count_split)) * self.expr_entropy)
    return [ExpressionArgs(op_count, entropy) for op_count, entropy in zip(count_split, entropies)]

def add_sub_filter(expr_value, expr_args):
  return expr_args.expr_count >= 2 or expr_value.is_Integer

def add_op(expr_value, expr_args, rationals_allowed):
  entropy, expr_args = expr_args.decompose()
  if rationals_allowed and expr_args.expr_count >= 3:
    x = num_sample.integer_or_rational(entropy, True)
  else:
    x = num_sample.integer(entropy, True)
  if random.choice([False, True]):
    op_args = [x, expr_value - x]
  else:
    op_args = [expr_value - x, x]
  return math_ops.Add, op_args, expr_args

def sub_op(expr_value, expr_args, rationals_allowed):
  entropy, expr_args = expr_args.decompose()
  if rationals_allowed and expr_args.expr_count >= 3:
    x = num_sample.integer_or_rational(entropy, True)
  else:
    x = num_sample.integer(entropy, True)
  if random.choice([False, True]):
    op_args = [x, x - expr_value]
  else:
    op_args = [expr_value + x, x]
  return math_ops.Sub, op_args, expr_args

def entropy_of_factor_split(integer):
  assert integer.is_Integer
  if integer == 0:
    return 0
  factors = sympy.factorint(integer)
  return sum(math.log10(mult + 1) for mult in six.itervalues(factors))

def split_factors(integer):
  assert integer.is_Integer
  if integer == 0:
    return [1, 0]
  factors = sympy.factorint(integer)
  left = sympy.Integer(1)
  right = sympy.Integer(1)
  for factor, mult in six.iteritems(factors):
    left_mult = random.randint(0, mult)
    right_mult = mult - left_mult
    left *= factor ** left_mult
    right *= factor ** right_mult
  return left, right

def mul_filter(expr_value, expr_args):
  if expr_args.expr_count >= 2:
    return True
  if not expr_value.is_Integer:
    return False
  return expr_args.expr_entropy <= entropy_of_factor_split(expr_value)

def mul_op(expr_value, expr_args, rationals_allowed):
  if expr_args.expr_count >= 3:
    _, op_args, expr_args = div_op(expr_value, expr_args, rationals_allowed)
    op_args = [op_args[0], sympy.Integer(1) / op_args[1]]
  elif expr_args.expr_count == 1:
    entropy, expr_args = expr_args.decompose()
    assert entropy_of_factor_split(expr_value) >= entropy
    op_args = split_factors(expr_value)
  else:
    assert expr_args.expr_count == 2
    entropy, expr_args = expr_args.decompose()
    numer = sympy.numer(expr_value)
    denom = sympy.denom(expr_value)
    p1, p2 = split_factors(numer)
    entropy -= entropy_of_factor_split(numer)
    mult = num_sample.integer(entropy, signed=True, min_abs=1, coprime_to=p1)
    op_args = [p1 / (mult * denom), p2 * mult]
  if random.choice([False, True]):
    op_args = list(reversed(op_args))
  return math_ops.Mul, op_args, expr_args

def div_filter(expr_value, expr_args):
  del expr_value
  del expr_args
  return True

def div_op(expr_value, expr_args, rationals_allowed):
  assert rationals_allowed
  entropy, expr_args = expr_args.decompose()
  numer = sympy.numer(expr_value)
  denom = sympy.denom(expr_value)
  if expr_args.expr_count == 1:
    mult = num_sample.integer(entropy, signed=True, min_abs=1)
    op_args = [numer * mult, denom * mult]
  elif expr_args.expr_count == 2:
    if numer == 0 or random.choice([False, True]):
      x = num_sample.integer(entropy, signed=True, min_abs=1, coprime_to=denom)
      op_args = [sympy.Rational(x * numer, denom), x]
    else:
      x = num_sample.integer(entropy, signed=True, min_abs=1, coprime_to=numer)
      op_args = [x, sympy.Rational(x * denom, numer)]
  else:
    assert expr_args.expr_count >= 3
    p2, p1 = split_factors(numer)
    q1, q2 = split_factors(denom)
    entropy -= entropy_of_factor_split(numer) + entropy_of_factor_split(denom)
    entropy_r = random.uniform(0, entropy)
    entropy_s = entropy - entropy_r
    r = num_sample.integer(entropy_r, signed=True, min_abs=1, coprime_to=q1*p2)
    s = num_sample.integer(entropy_s, signed=False, min_abs=1, coprime_to=p1*q2)
    op_args = [sympy.Rational(r*p1, s*q1), sympy.Rational(r*q2, s*p2)]
  return math_ops.Div, op_args, expr_args

def arithmetic(expr_value, expr_entropy, length=None, add_sub=True, mul_div=True):
  assert isinstance(expr_entropy, float)
  if length is None:
    min_length, max_length = length_range_for_entropy(expr_entropy)
    length = random.randint(min_length, max_length)
    expr_entropy -= math.log10(max_length - min_length + 1)
  else:
    assert isinstance(length, int)

  expr_entropy += combinatorics_util.log_number_binary_trees(length) / math.log(10)
  expr_value = sympy.sympify(expr_value)
  expr_args = ExpressionArgs(length, expr_entropy)
  return _arithmetic(expr_value, expr_args, add_sub, mul_div)

def _arithmetic(expr_value, expr_args, add_sub, mul_div):
  assert expr_args.expr_count >= 0
  if expr_args.expr_count == 0:
    assert expr_args.expr_entropy == 0
    return math_ops.Constant(expr_value)

  allowed = []
  if add_sub and add_sub_filter(expr_value, expr_args):
    allowed.append(add_op)
    allowed.append(sub_op)
  if mul_div and mul_filter(expr_value, expr_args):
    allowed.append(mul_op)
  if mul_div and div_filter(expr_value, expr_args):
    allowed.append(div_op)
  if not allowed:
    raise ValueError(
        'No valid ops found, add_sub={} mul_div={} expr_value={} expr_args={}'
        .format(add_sub, mul_div, expr_value, expr_args))
  choice = random.choice(allowed)

  op, args, expr_args = choice(expr_value, expr_args, rationals_allowed=mul_div)
  expr_args = expr_args.split(args)
  child_exprs = [_arithmetic(arg, child_expr_args, add_sub, mul_div)
                       for arg, child_expr_args in zip(args, expr_args)]

  return op(*child_exprs)

def length_range_for_entropy(expr_entropy):
  min_length = 3
  max_length = min_length + int(expr_entropy / 2)
  return min_length, max_length

def generate_expression(expr_value, expr_entropy, length=None, add_sub=True, mul_div=True):
  assert isinstance(expr_entropy, float)
  if length is None:
    min_length, max_length = length_range_for_entropy(expr_entropy)
    length = random.randint(min_length, max_length)
    expr_entropy -= math.log10(max_length - min_length + 1)
  else:
    assert isinstance(length, int)

  expr_entropy += combinatorics_util.log_number_binary_trees(length) / math.log(10)
  expr_value = sympy.sympify(expr_value)
  expr_args = ExpressionArgs(length, expr_entropy)
  return _arithmetic(expr_value, expr_args, add_sub, mul_div)
