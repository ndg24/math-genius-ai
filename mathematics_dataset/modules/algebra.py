from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random

from mathematics_dataset import example_math
from mathematics_dataset.sample import linear_system_math
from mathematics_dataset.sample import number_math
from mathematics_dataset.sample import ops_math
from mathematics_dataset.sample import polynomials_math
from mathematics_dataset.util import composition_math
from mathematics_dataset.util import display_math
import numpy as np
from six.moves import range as six_range
import sympy


_ENTROPY_TRAIN_MATH = (3, 10)
_ENTROPY_INTERPOLATE_MATH = (8, 8)
_ENTROPY_EXTRAPOLATE_MATH = (12, 12)

_POLY_PROBABILITY_REPEATED_ROOT_MATH = 0.2


def _make_math_modules(entropy):
    sample_args_pure = composition_math.PreSampleArgs(1, 1, *entropy)
    sample_args_composed = composition_math.PreSampleArgs(2, 4, *entropy)

    return {
        'polynomial_roots': functools.partial(
            polynomial_roots_math, None, sample_args_pure),
        'polynomial_roots_composed': functools.partial(
            polynomial_roots_math, None, sample_args_composed),
        'linear_1d': functools.partial(
            solve_linear_1d_math, None, sample_args_pure),
        'linear_1d_composed': functools.partial(
            solve_linear_1d_math, None, sample_args_composed),
        'linear_2d': functools.partial(
            solve_linear_2d_math, None, sample_args_pure),
        'linear_2d_composed': functools.partial(
            solve_linear_2d_math, None, sample_args_composed),
        'sequence_next_term': functools.partial(
            sequence_next_term_math, *entropy),
        'sequence_nth_term': functools.partial(
            sequence_nth_term_math, *entropy),
    }


def train_math(entropy_fn):
    return _make_math_modules(entropy_fn(_ENTROPY_TRAIN_MATH))


def test_math():
    return _make_math_modules(_ENTROPY_INTERPOLATE_MATH)


def test_extra_math():
    sample_args_pure = composition_math.PreSampleArgs(1, 1, *_ENTROPY_EXTRAPOLATE_MATH)
    return {
        'polynomial_roots_big': functools.partial(
            polynomial_roots_math, None, sample_args_pure),
    }


def _sample_roots_math(entropy):
    num_roots = random.randint(2, 5)

    num_repeated = np.random.binomial(
        num_roots - 1, _POLY_PROBABILITY_REPEATED_ROOT_MATH)
    if entropy > 4:
        num_repeated = min(num_repeated, int(num_roots / 2))

    num_distinct = num_roots - num_repeated

    entropies = entropy * np.random.dirichlet(np.ones(num_distinct))

    roots = []

    for root_entropy in entropies:
        if random.random() < 0.1:
            root = number_math.non_integer_rational(root_entropy, True)
        else:
            root = number_math.integer(root_entropy, True)
        roots.append(root)

    for _ in six_range(num_repeated):
        roots.append(random.choice(roots[:num_distinct]))

    return roots


def _polynomial_coeffs_with_roots_math(roots, scale_entropy):
    variable = sympy.Symbol('x')
    polynomial = sympy.Poly(sympy.prod([variable - root for root in roots]))
    coeffs_reversed = polynomial.all_coeffs()
    assert len(coeffs_reversed) == len(roots) + 1
    coeffs = list(reversed(coeffs_reversed))
    lcm = sympy.lcm([sympy.denom(coeff) for coeff in coeffs])
    if scale_entropy > 0:
        while True:
            scale = number_math.integer_or_rational(scale_entropy, signed=True)
            if scale != 0:
                break
    else:
        scale = 1
    return [coeff * scale * lcm for coeff in coeffs]


def polynomial_roots_math(value, sample_args, context=None):
    del value
    if context is None:
        context = composition_math.Context()

    entropy, sample_args = sample_args.peel()
    scale_entropy = min(entropy / 2, 1)

    roots = _sample_roots_math(entropy - scale_entropy)
    solutions = sorted(list(sympy.FiniteSet(*roots)))
    coeffs = _polynomial_coeffs_with_roots_math(roots, scale_entropy)
    (polynomial_entity,) = context.sample(
        sample_args, [composition_math.Polynomial(coeffs)])

    if random.choice([False, True]):
        if len(solutions) == 1:
            answer = solutions[0]
        else:
            answer = display_math.NumberList(solutions)

        if polynomial_entity.has_expression():
            equality = ops_math.Eq(polynomial_entity.expression, 0)
            variable = polynomial_entity.polynomial_variables[0]
        else:
            variable = sympy.Symbol(context.pop())
            equality = ops_math.Eq(polynomial_entity.handle.apply(variable), 0)
        template = random.choice([
            'Let {equality}. What is {variable}?',
            'Let {equality}. Calculate {variable}.',
            'Suppose {equality}. What is {variable}?',
            'Suppose {equality}. Calculate {variable}.',
            'What is {variable} in {equality}?',
            'Solve {equality} for {variable}.',
            'Find {variable} such that {equality}.',
            'Find {variable}, given that {equality}.',
            'Determine {variable} so that {equality}.',
            'Determine {variable}, given that {equality}.',
            'Solve {equality}.'
        ])
        return example_math.ProblemMath(
            question=example_math.generate_question_math(
                context, template, equality=equality, variable=variable),
            answer=answer)
    else:
        if polynomial_entity.has_expression():
            expression = polynomial_entity.expression
            variable = polynomial_entity.polynomial_variables[0]
        else:
            variable = sympy.Symbol(context.pop())
            expression = polynomial_entity.handle.apply(variable)
        factored = sympy.factor(
            polynomials_math.coefficients_to_polynomial(coeffs, variable))
        template = random.choice([
            'Factor {expression}.',
        ])
        return example_math.ProblemMath(
            question=example_math.generate_question_math(
                context, template, expression=expression),
            answer=factored)


def _solve_linear_system_math(degree, value, sample_args, context=None):
    is_question = context is None
    if context is None:
        context = composition_math.Context()

    entropy, sample_args = sample_args.peel()

    solutions = []
    if value is not None:
        solutions.append(value)

    extra_solutions_needed = degree - len(solutions)
    if extra_solutions_needed > 0:
        entropies = (entropy / 4) * np.random.dirichlet()
        np.ones(extra_solutions_needed)
