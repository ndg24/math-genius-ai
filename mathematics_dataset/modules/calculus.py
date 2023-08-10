import functools
import math
import random

from mathematics_dataset import example
from mathematics_dataset.sample import polynomials
from mathematics_dataset.util import composition
from mathematics_dataset.util import display
import numpy as np
import sympy

_ENTROPY_TRAIN = (3, 10)
_ENTROPY_INTERPOLATE = (8, 8)

def _make_modules(entropy):
  sample_args_pure = composition.PreSampleArgs(1, 1, *entropy)
  sample_args_composed = composition.PreSampleArgs(2, 4, *entropy)

  return {
      'diff_composed': functools.partial(
          differentiate_univariate, None, sample_args_composed),
      'diff': functools.partial(differentiate, None, sample_args_pure),
  }

def train(entropy_fn):
  return _make_modules(entropy_fn(_ENTROPY_TRAIN))

def test():
  return _make_modules(_ENTROPY_INTERPOLATE)

def test_extra():
  return {
  }

def _generate_polynomial(num_variables, entropy, derivative_order,
                         derivative_axis):
  degrees = np.random.randint(1, 4, [num_variables])
  degrees[derivative_axis] = np.random.randint(0, 4)

  coefficients = polynomials.sample_coefficients(degrees, entropy)
  assert derivative_order > 0
  degrees[derivative_axis] = derivative_order - 1
  extra_coefficients = polynomials.sample_coefficients(degrees, entropy)

  return np.concatenate(
      [extra_coefficients, coefficients], axis=derivative_axis)

def _template(module_count, derivative_order, num_variables):
  templates = [
      'Find the {nth} derivative of {eq} wrt {var}.',
      'What is the {nth} derivative of {eq} wrt {var}?',
  ]
  if derivative_order == 1:
    templates += [
        'Differentiate {eq} with respect to {var}.',
        'Differentiate {eq} wrt {var}.',
        'What is the derivative of {eq} wrt {var}?',
    ]

  derivative_variable_is_unambiguous = num_variables == 1 and module_count == 1
  if derivative_variable_is_unambiguous:
    templates += [
        'Find the {nth} derivative of {eq}.',
        'What is the {nth} derivative of {eq}?',
    ]
    if derivative_order == 1:
      templates += [
          'Differentiate {eq}.',
          'What is the derivative of {eq}?',
      ]

  return random.choice(templates)

def _sample_integrand(coefficients, derivative_order, derivative_axis, entropy):
  coefficients = np.asarray(coefficients)
  integrand = coefficients
  for _ in range(derivative_order):
    integrand = polynomials.integrate(integrand, derivative_axis)

  constant_degrees = np.array(integrand.shape) - 1
  constant_degrees[derivative_axis] = derivative_order - 1
  extra_coeffs = polynomials.sample_coefficients(constant_degrees, entropy)
  pad_amount = coefficients.shape[derivative_axis]
  pad = [(0, pad_amount if i == derivative_axis else 0)
         for i in range(coefficients.ndim)]
  extra_coeffs = np.pad(extra_coeffs, pad, 'constant', constant_values=0)
  return integrand + extra_coeffs

def _differentiate_polynomial(value, sample_args, context, num_variables):
  is_question = context is None
  if context is None:
    context = composition.Context()

  if value is not None:
    num_variables = value.coefficients.ndim

  entropy, sample_args = sample_args.peel()
  max_derivative_order = 3
  derivative_order = random.randint(1, max_derivative_order)
  entropy = max(0, entropy - math.log10(max_derivative_order))

  derivative_axis = random.randint(0, num_variables - 1)
  if value is None:
    coefficients = _generate_polynomial(
        num_variables, entropy, derivative_order, derivative_axis)
  else:
    coefficients = _sample_integrand(
        value.coefficients, derivative_order, derivative_axis, entropy)

  (entity,) = context.sample(
      sample_args, [composition.Polynomial(coefficients)])

  value = coefficients
  for _ in range(derivative_order):
    value = polynomials.differentiate(value, axis=derivative_axis)
  nth = display.StringOrdinal(derivative_order)

  if entity.has_expression():
    polynomial = entity.expression
    variables = entity.polynomial_variables
  else:
    variables = [sympy.Symbol(context.pop()) for _ in range(num_variables)]
    polynomial = entity.handle.apply(*variables)
  variable = variables[derivative_axis]

  if is_question:
    template = _template(context.module_count, derivative_order, len(variables))
    answer = polynomials.coefficients_to_polynomial(value, variables).sympy()
    return example.Problem(
        question=example.question(
            context, template, eq=polynomial, var=variable, nth=nth),
        answer=answer)
  else:
    fn_symbol = context.pop()
    variables_string = ', '.join(str(variable) for variable in variables)
    assert len(variables) == 1
    return composition.Entity(
        context=context,
        value=composition.Polynomial(value),
        description='Let {fn}({variables}) be the {nth} derivative of {eq}.',
        handle=composition.FunctionHandle(fn_symbol),
        fn=fn_symbol, variables=variables_string, nth=nth, eq=polynomial)

def differentiate_univariate(value, sample_args, context=None):
  return _differentiate_polynomial(value, sample_args, context, 1)

@composition.module(composition.is_polynomial)
def differentiate(value, sample_args, context=None):
  num_variables = random.randint(1, 4)
  return _differentiate_polynomial(value, sample_args, context, num_variables)
