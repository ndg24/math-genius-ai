from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import random

from mathematics_dataset import example
from mathematics_dataset.modules import train_test_split
from mathematics_dataset.sample import number
from mathematics_dataset.util import composition
from mathematics_dataset.util import display
import six
import sympy


def create_new_modules(is_training):
    return {
        'measurement_conversion': functools.partial(
            measurement_conversion, is_training=is_training, is_ext=False),
        'time_related': functools.partial(time_related, is_training=is_training),
    }


def training(entropy_func):
    del entropy_func
    return create_new_modules(is_training=True)


def testing():
    return create_new_modules(is_training=False)


def testing_extrapolation():
    return {
        'measurement_conversion': functools.partial(
            measurement_conversion, is_training=False, is_ext=True),
    }


Unit = collections.namedtuple('Unit', ('n', 's'))


MICRO_SYMBOL = 'u'


LENGTH = {
    Unit('m', 'm'): 1,
    Unit('km', 'km'): 1000,
    Unit('cm', 'cm'): sympy.Rational(1, 100),
    Unit('mm', 'mm'): sympy.Rational(1, 1000),
    Unit('um', 'um'): sympy.Rational(1, 1e6),
    Unit('nm', 'nm'): sympy.Rational(1, 1e9),
}

TIME = {
    Unit('s', 's'): 1,
    Unit('min', None): 60,
    Unit('h', None): 60*60,
    Unit('d', None): 24*60*60,
    Unit('wk', None): 7*24*60*60,
    Unit('ms', 'ms'): sympy.Rational(1, 1e3),
    Unit('us', MICRO_SYMBOL + 's'): sympy.Rational(1, 1e6),
    Unit('ns', 'ns'): sympy.Rational(1, 1e9),
}

TIME_YEARLY = {
    Unit('yr', None): 1,
    Unit('dec', None): 10,
    Unit('cen', None): 100,
    Unit('mill', None): 1000,
    Unit('mo', None): sympy.Rational(1, 12),
}

MASS = {
    Unit('kg', 'kg'): 1,
    Unit('t', 't'): 1000,
    Unit('g', 'g'): sympy.Rational(1, 1e3),
    Unit('mg', 'mg'): sympy.Rational(1, 1e6),
    Unit('ug', MICRO_SYMBOL + 'g'): sympy.Rational(1, 1e9),
    Unit('ng', 'ng'): sympy.Rational(1, 1e12),
}

VOLUME = {
    Unit('l', 'l'): 1,
    Unit('ml', 'ml'): sympy.Rational(1, 1000),
}


DIMENSIONS = [LENGTH, TIME, TIME_YEARLY, MASS, VOLUME]


def pluralize(name):
    if name == 'cen':
        return 'centuries'
    if name == 'mill':
        return 'millennia'
    return name + 's'


def factor_non_decimal(value):
    result = 1
    factors = sympy.factorint(value)
    for factor, power in six.iteritems(factors):
        if factor not in [2, 5]:
            result *= factor ** power
    return result


def sample_conversion_decimal(dimension, is_ext):
    base_unit, target_unit = random.sample(list(dimension.keys()), 2)
    scale = sympy.Rational(dimension[base_unit]) / dimension[target_unit]
    scale_non_decimal = factor_non_decimal(sympy.denom(scale))
    entropy = 9 if is_ext else 7
    base_value = number.non_integer_decimal(entropy, signed=False)
    base_value = display.Decimal(base_value.value * scale_non_decimal)
    target_value = display.Decimal(base_value.value * scale)
    return base_value, base_unit, target_value, target_unit


def conversion_decimal(context, is_training, is_ext):
    dimension = random.choice(DIMENSIONS)
    while True:
        base_value, base_unit, target_value, target_unit = (
            sample_conversion_decimal(dimension, is_ext))
        if train_test_split.is_train(base_value) == is_training:
            break

    templates = [
        'How many {t_name} are there in {b_value} {b_name}?',
        'What is {b_value} {b_name} in {t_name}?',
        'Convert {b_value} {b_name} to {t_name}.',
    ]
    if base_unit.s is not None:
        templates += [
            'How many {t_name} are there in {b_value}{b_symbol}?',
            'What is {b_value}{b_symbol} in {t_name}?',
            'Convert {b_value}{b_symbol} to {t_name}.',
        ]
    template = random.choice(templates)

    base_name = pluralize(base_unit.n)
    target_name = pluralize(target_unit.n)

    question = example.question(
        context,
        template,
        b_name=base_name,
        b_symbol=base_unit.s,
        b_value=base_value,
        t_name=target_name)
    return example.Problem(question=question, answer=target_value)


def conversion_fraction(context, is_training):
    dimension = random.choice(DIMENSIONS)
    allow_zero = random.random() < 0.2
    while True:
        base_unit, target_unit = random.sample(list(dimension.keys()), 2)
        base_value = number.non_integer_rational(2, signed=False)
        if train_test_split.is_train(base_value) != is_training:
            continue
        answer = (base_value * sympy.Rational(dimension[base_unit])
                  / sympy.Rational(dimension[target_unit]))
        if (abs(answer) <= 100000
                and sympy.denom(answer) == 1
                and (allow_zero or answer != 0)):
            break

    template = random.choice([
        'How many {t_name} are there in {b_value} of a {b_name}?',
        'What is {b_value} of a {b_name} in {t_name}?',
    ])

    if sympy.denom(base_value) > 20 or random.choice([False, True]):
        base_value_str = base_value
    else:
        base_value_str = display.StringNumber(base_value)

    question = example.question(
        context, template,
        b_name=base_unit.n,
        b_value=base_value_str,
        t_name=pluralize(target_unit.n))
    return example.Problem(question=question, answer=answer)


def measurement_conversion(is_training, is_ext):
    context = composition.Context()
    if is_ext or random.choice([False, True]):
        return conversion_decimal(
            context, is_training=is_training, is_ext=is_ext)
    else:
        return conversion_fraction(context, is_training=is_training)


def time_related(is_training):
    context = composition.Context()
    start_minutes = random.randint(1, 24*60 - 1)
    while True:
        duration_minutes = random.randint(1, 12*60 - 1)
        if train_test_split.is_train(duration_minutes) == is_training:
            break
    end_minutes = start_minutes + duration_minutes

    def format_12hr(minutes):
        hours = (minutes // 60) % 24
        minutes %= 60
        am_pm = 'AM' if hours < 12 else 'PM'
        hours = (hours - 1) % 12 + 1
        return '{}:{:02} {}'.format(hours, minutes, am_pm)

    start = format_12hr(start_minutes)
    end = format_12hr(end_minutes)

    which_question = random.randint(0, 3)
    if which_question == 0:
        template = random.choice([
            'What is {d} minutes before {e}?',
        ])
        return example.Problem(
            question=example.question(
                context, template, d=duration_minutes, e=end),
            answer=start)
    elif which_question == 1:
        template = random.choice([
            'What is {d} minutes after {s}?',
        ])
        return example.Problem(
            question=example.question(
                context, template, d=duration_minutes, s=start),
            answer=end)
    else:
        template = random.choice([
            'How many minutes are there between {s} and {e}?',
        ])
        return example.Problem(
            question=example.question(context, template, s=start, e=end),
            answer=duration_minutes)
