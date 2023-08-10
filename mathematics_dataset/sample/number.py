from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

from mathematics_dataset.util import display
import numpy as np
import six
import sympy

def _coprime_density(value):
  factors = sympy.factorint(value)
  density = 1.0
  for prime in six.iterkeys(factors):
    density *= 1 - 1 / prime
  return density

def integer(entropy, signed, min_abs=0, coprime_to=1):
  assert isinstance(min_abs, int) and not isinstance(min_abs, bool)
  coprime_to = abs(coprime_to)
  assert min_abs >= 0

  max_ = math.pow(10, entropy)
  max_ += min_abs
  if coprime_to >= 2:
    max_ = max_ / _coprime_density(coprime_to) + 1

  if signed:
    max_ = int(math.ceil(max_ / 2))
    range_ = [-max_, max_]
  else:
    max_ = int(math.ceil(max_))
    range_ = [min_abs, max_]

  while True:
    value = random.randint(*range_)
    if abs(value) >= min_abs and sympy.gcd(value, coprime_to) == 1:
      break

  return sympy.Integer(value)

def non_integer_rational(entropy, signed):
  numer_entropy = random.uniform(0, entropy)
  denom_entropy = entropy - numer_entropy
  numer = integer(numer_entropy, signed, min_abs=1)
  denom = integer(denom_entropy, False, min_abs=2, coprime_to=numer)
  return sympy.Rational(numer, denom)

def integer_or_rational(entropy, signed, min_abs=0):
  if random.choice([False, True]):
    return integer(entropy, signed, min_abs=min_abs)
  else:
    return non_integer_rational(entropy, signed)

def non_integer_decimal(entropy, signed):
  while True:
    base = integer(entropy, signed)
    shift = random.randint(1, int(math.ceil(entropy)))
    divisor = 10**shift
    if base % divisor != 0:
      return display.Decimal(sympy.Rational(base, divisor))

def integer_or_decimal(entropy, signed):
  if random.choice([False, True]):
    return display.Decimal(integer(entropy, signed))
  else:
    return non_integer_decimal(entropy, signed)

def entropy_of_value(value):
  if isinstance(value, display.Decimal):
    return entropy_of_value(sympy.numer(value))

  if is_non_integer_rational(value):
    numer = sympy.numer(value)
    denom = sympy.denom(value)
    return entropy_of_value(numer) + entropy_of_value(denom)
  elif not is_integer(value):
    raise ValueError('Unhandled value: {}'.format(value))

  return math.log10(5 * abs(value) + 1)

def is_integer(value):
  return isinstance(value, (int, np.int64, np.int32, sympy.Integer))

def is_positive_integer(value):
  return is_integer(value) and value > 0

def is_integer_or_rational(value):
  return is_integer(value) or isinstance(value, sympy.Rational)

def is_integer_or_decimal(value):
  return is_integer(value) or isinstance(value, display.Decimal)

def is_integer_or_rational_or_decimal(value):
  return is_integer_or_rational(value) or is_integer_or_decimal(value)

def is_non_integer_rational(value):
  return is_integer_or_rational(value) and not is_integer(value)
