from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from mathematics_dataset.util import combinatorics
import numpy as np
import six
import sympy
from sympy.solvers.diophantine import base_solution_linear as diophantine_solve_linear_2d

def expanded_coefficient_counts(length, is_zero):
  counts = np.asarray([0 if zero else 1 for zero in is_zero])
  extra_needed = (length - sum(counts))

  if extra_needed < 0:
    raise ValueError('length={} cannot handle is_zero={}'
                     .format(length, is_zero))

  extra = combinatorics.uniform_non_negative_integers_with_sum(
      count=len(is_zero), sum_=extra_needed)
  counts += np.asarray(extra)

  while True:
    bad_zeros = [
        i for i in range(len(is_zero)) if is_zero[i] and counts[i] == 1
    ]
    if not bad_zeros:
      break
    take_from = random.choice(bad_zeros)
    add_to = random.choice(
        [i for i in range(len(is_zero)) if counts[i] >= 1 and i != take_from])
    counts[take_from] -= 1
    counts[add_to] += 1

  return counts

def _split_value_equally(delta, count):
  numer = sympy.numer(delta)
  denom = sympy.denom(delta)
  return [int(math.floor((numer + i) / count)) / denom for i in range(count)]

def integers_with_sum(value, count, entropy):
  if count == 0:
    assert value == 0
    assert entropy == 0
    return []
  if count == 1:
    assert entropy == 0
    return [value]

  if not number.is_integer(value):
    raise ValueError('value={} (type={}) is not an integer'
                     .format(value, type(value)))

  entropy *= count / (count - 1)

  min_term_entropy = max(
      1, number.entropy_of_value(int(math.ceil(value/count))))
  term_entropies = entropy * np.random.dirichlet(np.ones(count))
  term_entropies = np.maximum(min_term_entropy, term_entropies)

  terms = [number.integer(term_entropy, signed=True)
           for term_entropy in term_entropies]

  delta = value - sum(terms)
  deltas = _split_value_equally(delta, count)
  terms = [term + delta for term, delta in zip(terms, deltas)]
  random.shuffle(terms)
  return terms

def monomial(coefficient, variables, powers):
  if not isinstance(variables, (list, tuple)):
    variables = [variables]
  if not isinstance(powers, (list, tuple, np.ndarray)):
    powers = [powers]

  terms = []

  for variable, power in zip(variables, powers):
    if power == 0:
      continue
    elif power == 1:
      terms.append(variable)
    else:
      terms.append(ops.Pow(variable, power))

  if (not terms
      or isinstance(coefficient, sympy.Symbol)
      or abs(coefficient) != 1):
    if isinstance(coefficient, sympy.Symbol):
      terms.insert(0, coefficient)
    else:
      terms.insert(0, abs(coefficient))

  if len(terms) > 1:
    term = ops.Mul(*terms)
  else:
    term = terms[0]

  if not isinstance(coefficient, sympy.Symbol) and coefficient < 0:
    term = ops.Neg(term)

  return term

def sample_coefficients(degrees, entropy, min_non_zero=0, max_non_zero=None):
  if isinstance(degrees, int):
    degrees = [degrees]
  degrees = np.asarray(degrees)

  def random_index():
    return [random.randint(0, degrees[i]) for i in range(len(degrees))]

  indices = set()
  for i, degree in enumerate(degrees):
    if degree > 0:
      index = random_index()
      index[i] = degree
      indices.add(tuple(index))

  abs_max_non_zero = np.prod(degrees + 1)

  min_non_zero = max(min_non_zero, 1, len(indices))
  if max_non_zero is None:
    max_non_zero = min_non_zero + int(entropy/2)

  min_non_zero = min(min_non_zero, abs_max_non_zero)
  max_non_zero = min(max_non_zero, abs_max_non_zero)
  max_non_zero = max(min_non_zero, max_non_zero)

  num_non_zero = random.randint(min_non_zero, max_non_zero)

  while len(indices) < num_non_zero:
    indices.add(tuple(random_index()))

  coeffs = np.zeros(degrees + 1, dtype=np.int64)
  entropies = entropy * np.random.dirichlet(np.ones(num_non_zero))

  for index, entry_entropy in zip(indices, entropies):
    value = number.integer(entry_entropy, signed=True, min_abs=1)
    coeffs.itemset(index, value)

  return coeffs

# ... (The code continues for other functions)

if __name__ == '__main__':
  absltest.main()

import numpy as np
import random
import math
import sympy
import six
import ops
import combinatorics
import number
import diophantine

def expand_coeffs(coeffs, entropy, length=None):
  coeffs = np.asarray(coeffs)
  shape = coeffs.shape

  expanded_coeffs = np.empty(shape, dtype=np.object)

  min_length = np.count_nonzero(coeffs) + 2
  if length is None:
    max_length = min_length + int(math.ceil(entropy) / 2)
    length = random.randint(min_length, max_length)
  if length < min_length:
    length = min_length

  is_zero_flat = np.reshape(coeffs, [-1]) == 0
  counts = expanded_coeff_counts(length, is_zero=is_zero_flat)
  coeffs_entropy = entropy * np.random.dirichlet(np.maximum(1e-9, counts - 1))
  counts = np.reshape(counts, shape)
  coeffs_entropy = np.reshape(coeffs_entropy, shape)

  indices = list(zip(*np.indices(shape).reshape([len(shape), -1])))
  for power in indices:
    coeffs = integers_with_sum(
        value=coeffs.item(power),
        count=counts.item(power),
        entropy=coeffs_entropy.item(power))
    expanded_coeffs.itemset(power, coeffs)

  return expanded_coeffs

def sample_expanded_coeffs(degrees, entropy, length=None):
  coeffs = sample_coeffs(degrees, entropy/2, max_non_zero=length)
  return expand_coeffs(coeffs, entropy/2, length)

def coeffs_to_polynomial(coeffs, variables):
  coeffs = np.asarray(coeffs)
  shape = coeffs.shape

  indices = list(zip(*np.indices(shape).reshape([len(shape), -1])))
  monomials = []
  for power in indices:
    c = coeffs.item(power)
    if (number.is_integer_or_rational(c) or isinstance(c, sympy.Symbol)):
      c = [c]
    elif not isinstance(c, list):
      raise ValueError('Unrecognized c={} type={}'.format(c, type(c)))
    for coeff in c:
      monomials.append(monomial(coeff, variables, power))
  random.shuffle(monomials)
  return ops.Add(*monomials)

def sample(variables, degrees, entropy, length=None):
  coeffs = sample_expanded_coeffs(degrees, entropy, length)
  return coeffs_to_polynomial(coeffs, variables)

def add_coeffs(c1, c2):
  c1 = np.asarray(c1)
  c2 = np.asarray(c2)

  d1 = np.array(c1.shape)
  d2 = np.array(c2.shape)
  assert len(d1) == len(d2)

  e1 = np.maximum(0, d2 - d1)
  e2 = np.maximum(0, d1 - d2)

  pad1 = [(0, e) for e in e1]
  pad2 = [(0, e) for e in e2]

  c1 = np.pad(c1, pad1, 'constant', constant_values=0)
  c2 = np.pad(c2, pad2, 'constant', constant_values=0)

  return c1 + c2

def _random_factor(n):
  factors = sympy.factorint(n)
  result = 1
  for f, p in six.iteritems(factors):
    result *= f ** random.randint(0, p)
  return result

def coeffs_linear_split(coeffs, entropy):
  coeffs = np.asarray(coeffs)
  shape = coeffs.shape
  coeffs = np.reshape(coeffs, [-1])

  entropy_a = max(1, random.uniform(0, entropy/3))
  entropy_b = max(1, random.uniform(0, entropy/3))
  entropy -= entropy_a + entropy_b
  entropy_coeffs = entropy * np.random.dirichlet(np.ones(len(coeffs)))

  coeffs_gcd = sympy.gcd([i for i in coeffs])
  coeffs_gcd = max(1, abs(coeffs_gcd))

  a = number.integer(entropy_a, signed=True, min_abs=1)
  b = number.integer(entropy_b, signed=True, min_abs=1, coprime_to=a)
  b *= _random_factor(coeffs_gcd)
  if random.choice([False, True]):
    a, b = b, a

  coeffs_1 = np.zeros(coeffs.shape, dtype=np.object)
  coeffs_2 = np.zeros(coeffs.shape, dtype=np.object)

  for index, coeff in enumerate(coeffs):
    entropy_coeff = entropy_coeffs[index]
    t = number.integer(entropy_coeff, signed=True)
    x, y = diophantine.solve_linear_2d(c=coeff, a=a, b=b, t=t)
    coeffs_1[index] = x
    coeffs_2[index] = y

  while np.all(coeffs_1 == 0) or np.all(coeffs_2 == 0):
    index = random.randint(0, len(coeffs) - 1)
    scale = random.choice([-1, 1])
    coeffs_1[index] += scale * b
    coeffs_2[index] -= scale * a

  coeffs_1 = np.reshape(coeffs_1, shape)
  coeffs_2 = np.reshape(coeffs_2, shape)

  return a, b, coeffs_1, coeffs_2

def _degree_of_variable(polynomial, variable):
  polynomial = sympy.sympify(polynomial).expand()
  if polynomial.is_constant():
    return 0
  polynomial = sympy.poly(polynomial)
  if variable not in polynomial.free_symbols:
    return 0
  return polynomial.degree(variable)

def _sample_with_brackets(depth, variables, degrees, entropy, length,
                          force_brackets=True):
  if force_brackets:
    length = max(2, length)

  if not force_brackets and (random.choice([False, True]) or length < 2):
    return sample(variables, degrees, entropy, length)

  length_left = random.randint(1, length - 1)
  length_right = length - length_left
  entropy_left, entropy_right = entropy * np.random.dirichlet(
      [length_left, length_right])

  if random.choice([False, True]):
    while True:
      left = _sample_with_brackets(
          depth + 1, variables, degrees, entropy_left, length_left, True)
      right = _sample_with_brackets(
          depth + 1, variables, degrees, entropy_right, length_right, False)
      if random.choice([False, True]):
        left, right = right, left
      result = ops.Add(left, right)
      all_ok = True
      for variable, degree in zip(variables, degrees):
        if _degree_of_variable(result, variable) != degree:
          all_ok = False
          break
      if all_ok:
        return result
  else:
    def sample_with_zero_check(degrees_, entropy_, length_):
      while True:
        result = _sample_with_brackets(
            depth + 1, variables, degrees_, entropy_, length_, False)
        if degrees_.sum() > 0 or not result.sympy().is_zero:
          return result
    degrees = np.asarray(degrees)

    def sample_degree(max_degree):
      if max_degree <= 1 or random.choice([False, True]):
        return random.randint(0, max_degree)
      return random.randint(1, max_degree - 1)

    degrees_left = np.array([sample_degree(degree) for degree in degrees])
    degrees_right = degrees - degrees_left
    left = sample_with_zero_check(degrees_left, entropy_left, length_left)
    right = sample_with_zero_check(degrees_right, entropy_right, length_right)
    return ops.Mul(left, right)

def sample_with_brackets(variables, degrees, entropy, length=None):
  if isinstance(degrees, int):
    degrees = [degrees]
  if not isinstance(variables, (list, tuple)):
    variables = [variables]

  if length is None:
    length = 3 + random.randint(0, int(entropy/2))

  entropy += combinatorics.log_number_binary_trees(length) / math.log(10)

  return _sample_with_brackets(0, variables, degrees, entropy, length, True)

def sample_with_small_evaluation(variable, degree, max_abs_input, entropy):
  assert max_abs_input >= 1
  entropies = entropy * np.random.dirichlet(np.ones(degree + 1))
  coeffs = []

  for power in range(degree + 1):
    delta = 0.5 * (degree - 2 * power) * math.log10(max_abs_input)
    power_entropy = entropies[power] + delta
    min_abs = 1 if power == degree else 0
    coeff = number.integer(power_entropy, signed=True, min_abs=min_abs)
    coeffs.append(coeff)

  terms = [monomial(coeff, variable, power)
           for power, coeff in enumerate(coeffs)]
  return ops.Add(*terms)

def sample_messy_power(variable, entropy):
  if entropy <= 0:
    return variable

  which = random.choice([1, 2, 3])

  if which == 1:
    exponent_entropy = min(2, entropy)
    entropy -= exponent_entropy
    exponent = number.integer_or_rational(exponent_entropy, signed=True)
    left = sample_messy_power(variable, entropy)
    return ops.Pow(left, exponent)

  entropy_left = entropy / 2
  if entropy_left < 1:
    entropy_left = 0
  entropy_right = entropy - entropy_left
  if random.choice([False, True]):
    entropy_left, entropy_right = entropy_right, entropy_left

  left = sample_messy_power(variable, entropy_left)
  right = sample_messy_power(variable, entropy_right)
  if which == 2:
    return ops.Mul(left, right)
  else:
    return ops.Div(left, right)

def trim(coeffs):
  coeffs = np.asarray(coeffs)
  non_zero = np.not_equal(coeffs, 0)
  ndim = coeffs.ndim

  for axis in range(ndim):
    length = coeffs.shape[axis]
    axis_complement = list(range(0, axis)) + list(range(axis + 1, ndim))
    non_zero_along_axis = np.any(non_zero, axis=tuple(axis_complement))

    slice_to = 0
    for index in range(length - 1, -1, -1):
      if non_zero_along_axis[index]:
        slice_to = index + 1
        break

    if slice_to < length:
      coeffs = coeffs.take(axis=axis, indices=list(range(slice_to)))

  return coeffs

def differentiate(coeffs, axis):
  coeffs = np.asarray(coeffs)
  indices = list(range(1, coeffs.shape[axis]))
  coeffs = coeffs.take(axis=axis, indices=indices)

  broadcast_shape = np.ones(coeffs.ndim, dtype=np.int32)
  broadcast_shape[axis] = len(indices)
  broadcast = np.asarray(indices).reshape(broadcast_shape)

  result = broadcast * coeffs
  return trim(result)

def integrate(coeffs, axis):
  coeffs = np.asarray(coeffs)

  length = coeffs.shape[axis]
  broadcast_shape = np.ones(coeffs.ndim, dtype=np.int32)
  broadcast_shape[axis] = length
  powers = np.array([sympy.Integer(i) for i in range(1, length + 1)])
  powers = powers.reshape(broadcast_shape)

  result_unpadded = coeffs / powers

  pad = [(1 if i == axis else 0, 0) for i in range(coeffs.ndim)]
  return np.pad(result_unpadded, pad, 'constant', constant_values=0)
