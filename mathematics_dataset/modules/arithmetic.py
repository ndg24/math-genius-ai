from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import random

from mathematics_dataset import example
from mathematics_dataset.sample import arithmetic
from mathematics_dataset.sample import number
from mathematics_dataset.sample import ops
from mathematics_dataset.util import composition
from mathematics_dataset.util import display
import sympy


_ENTROPY_TRAIN = (3, 10)
_ENTROPY_INTERPOLATE = (8, 8)
_ENTROPY_EXTRAPOLATE = (10, 12)

_ADD_SUB_ENTROPY_TRAIN = (4, 16)
_ADD_SUB_ENTROPY_INTERPOLATE = (12, 12)
_ADD_SUB_ENTROPY_EXTRAPOLATE = (16, 20)

_EXTRAPOLATE_EXTRA_LENGTH = 3

_INT = 'int'
_INT_OR_RATIONAL = 'rational'


def _make_modules(entropy, add_sub_entropy):
    sample_args_pure = composition.PreSampleArgs(1, 1, *entropy)
    add_sub_sample_args_pure = composition.PreSampleArgs(1, 1, *add_sub_entropy)
    
    return {
        'add_or_sub': functools.partial(add_or_sub, None, add_sub_sample_args_pure),
        'add_sub_multiple': functools.partial(add_sub_multiple, _INT, sample_args_pure),
        'add_or_sub_in_base': functools.partial(add_or_sub_in_base, sample_args_pure),
        'mul': functools.partial(mul, None, sample_args_pure),
        'div': functools.partial(div, None, sample_args_pure),
        'mul_div_multiple': functools.partial(mul_div_multiple, _INT_OR_RATIONAL, sample_args_pure),
        'mixed': functools.partial(mixed, _INT_OR_RATIONAL, sample_args_pure),
        'nearest_integer_root': functools.partial(nearest_integer_root, sample_args_pure),
        'simplify_surd': functools.partial(simplify_surd, None, sample_args_pure),
    }


def train(entropy_fn):
    return _make_modules(
        entropy=entropy_fn(_ENTROPY_TRAIN),
        add_sub_entropy=entropy_fn(_ADD_SUB_ENTROPY_TRAIN))


def test():
    return _make_modules(
        entropy=_ENTROPY_INTERPOLATE,
        add_sub_entropy=_ADD_SUB_ENTROPY_INTERPOLATE)


def test_extra():
    sample_args_pure = composition.PreSampleArgs(1, 1, *_ENTROPY_EXTRAPOLATE)
    add_sub_sample_args_pure = composition.PreSampleArgs(1, 1, *_ADD_SUB_ENTROPY_EXTRAPOLATE)

    train_length = arithmetic.length_range_for_entropy(_ENTROPY_TRAIN[1])[1]

    def extrapolate_length():
        return random.randint(train_length + 1, train_length + _EXTRAPOLATE_EXTRA_LENGTH)

    def add_sub_multiple_longer():
        return add_sub_multiple(_INT, sample_args_pure, length=extrapolate_length())

    def mul_div_multiple_longer():
        return mul_div_multiple(_INT, sample_args_pure, length=extrapolate_length())

    def mixed_longer():
        return mixed(_INT, sample_args_pure, length=extrapolate_length())

    return {
        'add_or_sub_big': functools.partial(add_or_sub, None, add_sub_sample_args_pure),
        'mul_big': functools.partial(mul, None, sample_args_pure),
        'div_big': functools.partial(div, None, sample_args_pure),
        'add_sub_multiple_longer': add_sub_multiple_longer,
        'mul_div_multiple_longer': mul_div_multiple_longer,
        'mixed_longer': mixed_longer,
    }


def _value_sampler(value):
    if value == _INT or number.is_integer(value):
        return functools.partial(number.integer, signed=True)
    if value == _INT_OR_RATIONAL or isinstance(value, sympy.Rational):
        return functools.partial(number.integer_or_rational, signed=True)
    if isinstance(value, display.Decimal):
        return functools.partial(number.integer_or_decimal, signed=True)
    raise ValueError('Unrecognized value {} of type {}'.format(value, type(value)))


def _add_question_or_entity(context, p, q, is_question):
    value = p.value + q.value

    if is_question:
        template = random.choice([
            '{p} + {q}',
            '{p}+{q}',
            'Work out {p} + {q}.',
            'Add {p} and {q}.',
            'Put together {p} and {q}.',
            'Sum {p} and {q}.',
            'Total of {p} and {q}.',
            'Add together {p} and {q}.',
            'What is {p} plus {q}?',
            'Calculate {p} + {q}.',
            'What is {p} + {q}?',
        ])
        return example.Problem(
            question=example.question(context, template, p=p, q=q),
            answer=value)
    else:
        return composition.Entity(
            context=context,
            value=value,
            description='Let {self} = {p} + {q}.',
            p=p, q=q)


def _sub_question_or_entity(context, p, q, is_question):
    value = p.value - q.value

    if is_question:
        templates = [
            '{p} - {q}',
            'Work out {p} - {q}.',
            'What is {p} minus {q}?',
            'What is {p} take away {q}?',
            'What is {q} less than {p}?',
            'Subtract {q} from {p}.',
            'Calculate {p} - {q}.',
            'What is {p} - {q}?',
        ]
        if sympy.Ge(p.value, q.value):
            for adjective in ['distance', 'difference']:
                for pair in ['{p} and {q}', '{q} and {p}']:
                    templates.append('What is the {} between {}?'.format(adjective, pair))
        template = random.choice(templates)
        return example.Problem(
            question=example.question(context, template, p=p, q=q),
            answer=value)
    else:
        return composition.Entity(
            context=context,
            value=value,
            description='Let {self} = {p} - {q}.',
            p=p, q=q)


def _entropy_for_pair(entropy):
    entropy_1 = max(1, random.uniform(0, entropy))
    entropy_2 = max(1, entropy - entropy_1)
    return entropy_1, entropy_2


@composition.module(number.is_integer_or_rational_or_decimal)
def add_or_sub(value, sample_args, context=None):
    is_question = context is None
    if context is None:
        context = composition.Context()

    is_addition = random.choice([False, True])
    entropy, sample_args = sample_args.peel()

    if value is None:
        entropy_p, entropy_q = _entropy_for_pair(entropy)
        p = number.integer_or_decimal(entropy_p, signed=True)
        q = number.integer_or_decimal(entropy_q, signed=True)
    else:
        entropy = max(entropy, number.entropy_of_value(value))
        sampler = _value_sampler(value)
        p = sampler(entropy)
        if is_addition:
            q = value - p
            if random.choice([False, True]):
                p, q = q, p
        else:
            q = p - value
            if random.choice([False, True]):
                p, q = -q, -p

    p, q = context.sample(sample_args, [p, q])

    if is_addition:
        return _add_question_or_entity(context, p, q, is_question)
    else:
        return _sub_question_or_entity(context, p, q, is_question)


def add_or_sub_in_base(sample_args):
    context = composition.Context()
    entropy, sample_args = sample_args.peel()
    entropy_p, entropy_q = _entropy_for_pair(entropy)
    p = number.integer(entropy_p, signed=True)
    q = number.integer(entropy_q, signed=True)
    base = random.randint(2, 16)
    if random.choice([False, True]):
        answer = p + q
        template = 'In base {base}, what is {p} + {q}?'
    else:
        answer = p - q
        template = 'In base {base}, what is {p} - {q}?'
    return example.Problem(
        question=example.question(
            context,
            template,
            base=base,
            p=display.NumberInBase(p, base),
            q=display.NumberInBase(q, base)),
        answer=display.NumberInBase(answer, base))


def mul(value, sample_args, context=None):
    del value
    is_question = context is None
    if context is None:
        context = composition.Context()
    entropy, sample_args = sample_args.peel()
    entropy_p, entropy_q = _entropy_for_pair(entropy)
    p = number.integer_or_decimal(entropy_p, True)
    q = number.integer_or_decimal(entropy_q, True)
    p, q = context.sample(sample_args, [p, q])
    answer = p.value * q.value

    if is_question:
        templates = [
            '{p}' + ops.MUL_SYMBOL + '{q}',
            '{p} ' + ops.MUL_SYMBOL + ' {q}',
            'Calculate {p}' + ops.MUL_SYMBOL + '{q}.',
            'Work out {p} ' + ops.MUL_SYMBOL + ' {q}.',
            'Multiply {p} and {q}.',
            'Product of {p} and {q}.',
            'What is the product of {p} and {q}?',
            '{p} times {q}',
            'What is {p} times {q}?',
        ]
        template = random.choice(templates)
        return example.Problem(
            question=example.question(context, template, p=p, q=q),
            answer=answer
        )
    else:
        return composition.Entity(
            context=context,
            value=answer,
            description='Let {self} = {p} * {q}.',
            p=p, q=q)


def div(value, sample_args, context=None):
    del value
    is_question = context is None
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()
    entropy_1, entropy_q = _entropy_for_pair(entropy)

    q = number.integer(entropy_q, True, min_abs=1)

    if random.choice([False, True]):
        answer = number.integer(entropy_1, True)
        p = answer * q
    else:
        p = number.integer(entropy_1, True)
        answer = p / q

    p, q = context.sample(sample_args, [p, q])

    if is_question:
        template = random.choice([
            'Divide {p} by {q}.',
            '{p} divided by {q}',
            'What is {p} divided by {q}?',
            'Calculate {p} divided by {q}.',
        ])
        return example.Problem(
            question=example.question(context, template, p=p, q=q),
            answer=answer
        )
    else:
        return composition.Entity(
            context=context,
            value=answer,
            description='Let {self} be {p} divided by {q}.',
            p=p, q=q)


def nearest_integer_root(sample_args):
    del sample_args
    context = composition.Context()

    if random.choice([False, True]):
        one_over_exponent = random.randint(2, 3)
    else:
        one_over_exponent = random.randint(2, 10)

    entropy, _ = sample_args.peel()
    value = number.integer(entropy, signed=False)
    answer = int(round(value ** (1 / one_over_exponent)))

    templates = [
        'What is {value} to the power of 1/{one_over_exponent}, to the nearest integer?',
    ]

    if one_over_exponent != 2:
        ordinal = str()
        templates += [
            'What is the {ordinal} root of {value} to the nearest integer?',
        ]

    if one_over_exponent == 2:
        templates += [
            'What is the square root of {value} to the nearest integer?',
        ]
    elif one_over_exponent == 3:
        templates += [
            'What is the cube root of {value} to the nearest integer?',
        ]

    template = random.choice(templates)

    ordinal = display.StringOrdinal(one_over_exponent)
    return example.Problem(
        question=example.question(
            context, template, value=value, ordinal=ordinal,
            one_over_exponent=one_over_exponent),
        answer=answer)


def _calculate(value, sample_args, context, add_sub, mul_div, length=None):
    is_question = context is None
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()

    if value in [_INT, _INT_OR_RATIONAL]:
        value_entropy = max(1.0, entropy / 4)
        entropy = max(1.0, entropy - value_entropy)
        sampler = _value_sampler(value)
        value = sampler(value_entropy)

    op = arithmetic.arithmetic(
        value=value, entropy=entropy, add_sub=add_sub, mul_div=mul_div,
        length=length)
    context.sample_by_replacing_constants(sample_args, op)

    if is_question:
        template = random.choice([
            '{op}',
            'What is {op}?',
            'Evaluate {op}.',
            'Calculate {op}.',
            'What is the value of {op}?',
        ])
        return example.Problem(
            question=example.question(context, template, op=op),
            answer=value)
    else:
        return composition.Entity(
            context=context,
            value=value,
            expression=op,
            description='Let {self} be {op}.',
            op=op)


def add_sub_multiple(value, sample_args, length=None):
    return _calculate(
        value, sample_args, None, add_sub=True, mul_div=False, length=length)


def mul_div_multiple(value, sample_args, length=None):
    return _calculate(
        value, sample_args, None, add_sub=False, mul_div=True, length=length)


@composition.module(number.is_integer_or_rational)
def mixed(value, sample_args, context=None, length=None):
    return _calculate(
        value, sample_args, context, add_sub=True, mul_div=True, length=length)


def _surd_coefficients(sympy_exp):
    sympy_exp = sympy.simplify(sympy.expand(sympy_exp))

    def extract_b(b_sqrt_base):
        if isinstance(b_sqrt_base, sympy.Pow):
            return 1
        else:
            assert isinstance(b_sqrt_base, sympy.Mul)
            assert len(b_sqrt_base.args) == 2
            assert b_sqrt_base.args[0].is_rational
            assert isinstance(b_sqrt_base.args[1], sympy.Pow)
            return b_sqrt_base.args[0]

    if sympy_exp.is_rational:
        return sympy_exp, 0
    elif isinstance(sympy_exp, sympy.Add):
        assert len(sympy_exp.args) == 2
        assert sympy_exp.args[0].is_rational
        a = sympy_exp.args[0]
        b = extract_b(sympy_exp.args[1])
        return a, b
    else:
        return 0, extract_b(sympy_exp)


def _surd_split_entropy_two(entropy):
    entropy_left = entropy / 2
    if entropy_left < 1:
        entropy_left = 0
    entropy_right = entropy - entropy_left
    if random.choice([False, True]):
        entropy_left, entropy_right = entropy_right, entropy_left
    return entropy_left, entropy_right


def _sample_surd(base, entropy, max_power, multiples_only):
    if entropy <= 0:
        return ops.Sqrt(base)

    def add_or_sub_():
        entropy_left, entropy_right = _surd_split_entropy_two(entropy)
        left = _sample_surd(base, entropy_left, max_power, multiples_only)
        right = _sample_surd(base, entropy_right, max_power, multiples_only)
        op = random.choice([ops.Add, ops.Sub])
        return op(left, right)

    def mul_by_integer():
        entropy_k = min(1, entropy)
        left = number.integer(entropy_k, signed=True, min_abs=1)
        right = _sample_surd(base, entropy - entropy_k, max_power, multiples_only)
        if random.choice([False, True]):
            left, right = right, left
        return ops.Mul(left, right)

    def div_by_sqrt_k():
        entropy_k = min(1, entropy)
        k = number.integer(entropy_k, signed=False, min_abs=2)
        entropy_left, entropy_right = _surd_split_entropy_two(entropy - entropy_k)
        k_base_expr = _sample_surd(k * base, entropy_left, max_power, True)
        while True:
            k_expr = _sample_surd(k, entropy_right, max_power, True)
            if k_expr.sympy() != 0:
                break
        return ops.Div(k_base_expr, k_expr)

    def square_k():
        entropy_k = min(1, entropy)
        k = number.integer(entropy_k, signed=False, min_abs=2)
        return _sample_surd(
            k * k * base, entropy - entropy_k, max_power, multiples_only)

    def surd_plus_integer():
        entropy_k = min(1, entropy)
        left = number.integer(entropy_k, signed=True)
        assert not multiples_only
        right = _sample_surd(base, entropy - entropy_k, max_power, False)
        if random.choice([True, False]):
            left, right = right, left
        return ops.Add(left, right)

    def power():
        assert not multiples_only
        surd = _sample_surd(base, entropy, max_power=1, multiples_only=False)
        return ops.Pow(surd, 2)

    choices = [add_or_sub_, mul_by_integer]
    if not multiples_only:
        choices += [surd_plus_integer]
        if max_power > 1:
            choices += [power]
    if base < 64:
        choices += [div_by_sqrt_k, square_k]
    which = random.choice(choices)
    return which()


def simplify_surd(value, sample_args, context=None):
    del value
    if context is None:
        context = composition.Context()

    entropy, sample_args = sample_args.peel()

    while True:
        base = random.randint(2, 20)
        if sympy.Integer(base).is_prime:
            break
    num_primes_less_than_20 = 8
    entropy -= math.log10(num_primes_less_than_20)
    exp = _sample_surd(base, entropy, max_power=2, multiples_only=False)
    simplified = sympy.expand(sympy.simplify(exp))

    template = random.choice([
        'Simplify {exp}.',
    ])
    return example.Problem(
        question=example.question(context, template, exp=exp),
        answer=simplified)
