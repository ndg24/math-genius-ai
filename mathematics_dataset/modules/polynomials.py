import functools
import math
import random

from mathematics_dataset import example
from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from mathematics_dataset.sample import polynomials
from mathematics_dataset.util import composition
import numpy as np
from six.moves import range
import sympy

ENTROPY_TRAIN = (3, 10)
ENTROPY_INTERPOLATE = (8, 8)

def make_new_modules(entropy):
  sample_args_pure = composition.PreSampleArgs(1, 1, *entropy)
  sample_args_composed = composition.PreSampleArgs(2, 4, *entropy)
  sample_args_mixed = composition.PreSampleArgs(1, 4, *entropy)

  return {
      'coefficient_named':
          functools.partial(coefficient_named, None, sample_args_pure),
      'evaluate':
          functools.partial(evaluate, None, sample_args_pure),
      'evaluate_composed':
          functools.partial(evaluate, None, sample_args_composed),
      'add':
          functools.partial(add, None, sample_args_mixed),
      'expand':
          functools.partial(expand, None, sample_args_pure),
      'collect':
          functools.partial(collect, None, sample_args_pure),
      'compose':
          functools.partial(compose, None, sample_args_mixed),
      'simplify_power':
          functools.partial(simplify_power, None, sample_args_pure),
  }

def train(entropy_fn):
  return make_new_modules(entropy_fn(ENTROPY_TRAIN))

def test():
  return make_new_modules(ENTROPY_INTERPOLATE)

def test_extra():
  return {
  }

def coefficient_named(val, sample_args, context=None):
  if context is None:
    context = composition.Context()
  var = sympy.Symbol(context.pop())

  entropy, sample_args = sample_args.peel()
  degree = random.randint(1, 4)
  if random.choice([False, True]):
    coeffs = polynomials.sample_coefficients(
        degree, entropy/2, min_non_zero=random.randint(degree - 1, degree))
    expanded = polynomials.expand_coefficients(coeffs, entropy/2)
    expression = polynomials.coefficients_to_polynomial(expanded, var)
  else:
    expression = polynomials.sample_with_brackets(var, degree, entropy)
    coeffs = list(reversed(sympy.Poly(expression).all_coeffs()))

  named_coeffs = [sympy.Symbol(context.pop()) for _ in range(degree + 1)]
  canonical = polynomials.coefficients_to_polynomial(named_coeffs, var)

  if random.random() < 0.2:
    power = random.randint(0, degree)
  else:
    non_zero_powers = [i for i in range(degree + 1) if coeffs[i] != 0]
    power = random.choice(non_zero_powers)

  value = coeffs[power]
  named_coeff = named_coeffs[power]

  template = random.choice([
      'Express {expression} as {canonical} and give {target}.',
      'Rearrange {expression} to {canonical} and give {target}.',
      'Express {expression} in the form {canonical} and give {target}.',
      'Rearrange {expression} to the form {canonical} and give {target}.',
  ])
  return example.Problem(
      example.question(context, template, expression=expression, canonical=canonical,
                       target=named_coeff),
      answer=value)

TEMPLATES = [
    'What is {composed}?',
    'Calculate {composed}.',
    'Give {composed}.',
    'Determine {composed}.',
]

@composition.module(number.is_integer)
def evaluate(val, sample_args, context=None):
  is_question = context is None
  if context is None:
    context = composition.Context()

  entropy, sample_args = sample_args.peel()

  if val is None:
    entropy_value = random.uniform(1, 1 + entropy/3)
    entropy = max(0, entropy - entropy_value)
    val = number.integer(entropy_value, signed=True)

  entropy_input = random.uniform(1, 1 + entropy/3)
  entropy = max(0, entropy - entropy_input)
  input_ = number.integer(entropy_input, signed=True)

  degree = random.randint(1, 3)

  entropies = entropy * np.random.dirichlet(list(range(1, degree + 1)))
  target = val
  coeffs_reversed = []
  for i, coeff_entropy in enumerate(entropies):
    power = degree - i
    coeff = number.integer(coeff_entropy, signed=True)
    if input_ != 0:
      coeff += int(round(target / input_ ** power))
    if coeff == 0 and i == 0:
      coeff += random.choice([-1, 1])
    coeffs_reversed.append(coeff)
    target -= coeff * (input_ ** power)
  coeffs_reversed.append(target)

  coeffs = list(reversed(coeffs_reversed))

  (polynomial_entity, input_) = context.sample(
      sample_args, [composition.Polynomial(coeffs), input_])
  composed = polynomial_entity.handle.apply(input_.handle)

  if is_question:
    template = random.choice(TEMPLATES)
    return example.Problem(
        example.question(context, template, composed=composed),
        answer=val)
  else:
    return composition.Entity(
        context=context,
        value=val,
        expression=composed,
        description='Let {self} be {composed}.',
        composed=composed)

def add(val, sample_args, context=None):
  is_question = context is None
  if context is None:
    context = composition.Context()

  entropy, sample_args = sample_args.peel()

  if val is None:
    max_degree = 3
    degree = random.randint(1, max_degree)
    entropy -= math.log10(max_degree)
    entropy_value = entropy / 2
    entropy -= entropy_value
    val = polynomials.sample_coefficients(
        degree, entropy=entropy_value, min_non_zero=random.randint(1, 3))
    val = composition.Polynomial(val)

  c1, c2, coeffs1, coeffs2 = polynomials.coefficients_linear_split(
      val.coefficients, entropy)
  coeffs1 = polynomials.trim(coeffs1)
  coeffs2 = polynomials.trim(coeffs2)

  c1, c2, fn1, fn2 = context.sample(
      sample_args,
      [c1, c2, composition.Polynomial(coeffs1), composition.Polynomial(coeffs2)]
  )

  var = sympy.Symbol(context.pop())

  expression = (
      c1.handle * fn1.handle.apply(var) + c2.handle * fn2.handle.apply(var))

  if is_question:
    answer = polynomials.coefficients_to_polynomial(val.coefficients, var)
    answer = answer.sympy()
    template = random.choice(TEMPLATES)
    return example.Problem(
        example.question(context, template, composed=expression),
        answer=answer)
  else:
    intermediate_symbol = context.pop()
    intermediate = sympy.Function(intermediate_symbol)(var)
    return composition.Entity(
        context=context,
        value=val,
        description='Let {intermediate} = {composed}.',
        handle=composition.FunctionHandle(intermediate_symbol),
        intermediate=intermediate,
        composed=expression)

def expand(val, sample_args, context=None):
  if context is None:
    context = composition.Context()
  var = sympy.Symbol(context.pop())
  entropy, sample_args = sample_args.peel()

  min_order = 1
  max_order = 5
  order = random.randint(min_order, max_order)
  entropy -= math.log10(max_order - min_order + 1)
  expression_ = polynomials.sample_with_brackets(var, order, entropy)
  expanded = sympy.expand(expression_)
  template = random.choice([
      'Expand {expression}.'
  ])
  return example.Problem(
      example.question(context, template, expression=expression_),
      answer=expanded)

@composition.module(composition.is_polynomial)
def collect(val, sample_args, context=None):
  is_question = context is None
  if context is None:
    context = composition.Context()

  entropy, sample_args = sample_args.peel()
  if val is None:
    entropy_value, entropy = entropy * np.random.dirichlet([2, 3])
    degrees = [random.randint(1, 3)]
    val = composition.Polynomial(
        polynomials.sample_coefficients(degrees, entropy_value))

  assert isinstance(val, composition.Polynomial)
  coeffs = val.coefficients

  all_coefficients_are_integer = True
  for coeff in coeffs.flat:
    if not number.is_integer(coeff):
      all_coefficients_are_integer = False
      break

  if all_coefficients_are_integer:
    coeffs = polynomials.expand_coefficients(coeffs, entropy)
  else:
    sample_args = composition.SampleArgs(
        sample_args.num_modules, sample_args.entropy + entropy)

  num_variables = coeffs.ndim
  variables = [sympy.Symbol(context.pop()) for _ in range(num_variables)]
  unsimplified = polynomials.coefficients_to_polynomial(coeffs, variables)
  simplified = unsimplified.sympy().expand()

  if not ops.number_constants(unsimplified):
    unsimplified = ops.Add(unsimplified, ops.Constant(0))
  context.sample_by_replacing_constants(sample_args, unsimplified)

  if is_question:
    template = 'Collect the terms in {unsimplified}.'
    return example.Problem(
        example.question(context, template, unsimplified=unsimplified),
        answer=simplified)
  else:
    function_symbol = context.pop()
    function = sympy.Function(function_symbol)(*variables)
    return composition.Entity(
        context=context,
        value=val,
        handle=composition.FunctionHandle(function_symbol),
        expression=unsimplified,
        polynomial_variables=variables,
        description='Let {function} = {unsimplified}.',
        function=function,
        unsimplified=unsimplified)

def compose(val, sample_args, context=None):
  if context is None:
    context = composition.Context()

  entropy, sample_args = sample_args.peel()
  entropy_f, entropy_g = entropy * np.random.dirichlet([1, 1])

  coeffs_f = polynomials.sample_coefficients([random.randint(1, 2)], entropy_f)
  coeffs_g = polynomials.sample_coefficients([random.randint(1, 2)], entropy_g)

  entity_f, entity_g = context.sample(
      sample_args,
      [composition.Polynomial(coeffs_f), composition.Polynomial(coeffs_g)])

  var = sympy.var(context.pop())

  poly_f = polynomials.coefficients_to_polynomial(coeffs_f, var)
  poly_g = polynomials.coefficients_to_polynomial(coeffs_g, var)

  poly_f_g = poly_f.sympy().subs(var, poly_g.sympy()).expand()

  expression = composition.FunctionHandle(entity_f, entity_g).apply(var)

  template = random.choice(TEMPLATES)
  return example.Problem(
      example.question(context, template, composed=expression),
      answer=poly_f_g)

def simplify_power(val, sample_args, context=None):
  if context is None:
    context = composition.Context()

  entropy, sample_args = sample_args.peel()

  var = sympy.symbols(context.pop(), positive=True)
  unsimplified = polynomials.sample_messy_power(var, entropy)
  answer = unsimplified.sympy()

  template = random.choice([
      'Simplify {unsimplified} assuming {variable} is positive.',
  ])
  return example.Problem(
      example.question(
          context, template, unsimplified=unsimplified, variable=var),
      answer=answer)
