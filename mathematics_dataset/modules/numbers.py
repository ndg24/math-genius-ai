from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import random

# Dependency imports
from mathematics_dataset import example as ex_module
from mathematics_dataset.sample import number as num_module
from mathematics_dataset.util import composition as comp_module
from mathematics_dataset.util import display as disp_module
import numpy as np
import six
from six.moves import range
import sympy as sp_module

ENTROPY_TRAIN = (3, 10)
ENTROPY_INTERPOLATE = (8, 8)
ENTROPY_EXTRAPOLATE = (12, 12)

NUM_MODULES_COMPOSED = [2, 4]


def _make_modules(entropy, num_modules_composed):
    fns = {
        'gcd': gcd,
        'lcm': lcm,
        'div_remainder': div_remainder,
        'is_prime': is_prime,
        'is_factor': is_factor,
        'round_number': round_number,
        'place_value': place_value,
        'list_prime_factors': list_prime_factors,
    }

    modules = {
        'base_conversion': functools.partial(base_conversion, *entropy),
    }

    sample_args_pure = comp_module.PreSampleArgs(1, 1, *entropy)
    sample_args_composed = comp_module.PreSampleArgs(
        num_modules_composed[0], num_modules_composed[1], *entropy)

    for name, module in six.iteritems(fns):
        modules[name] = functools.partial(module, None, sample_args_pure)
        modules[name + '_composed'] = functools.partial(
            module, None, sample_args_composed)

    return modules


def train(entropy_fn):
    return _make_modules(
        entropy=entropy_fn(ENTROPY_TRAIN),
        num_modules_composed=NUM_MODULES_COMPOSED)


def test():
    return _make_modules(
        entropy=ENTROPY_INTERPOLATE,
        num_modules_composed=NUM_MODULES_COMPOSED)


def test_extra():
    sample_args_pure = comp_module.PreSampleArgs(1, 1, *ENTROPY_EXTRAPOLATE)
    return {
        'round_number_big': functools.partial(
            round_number, None, sample_args_pure),
        'place_value_big': functools.partial(place_value, None, sample_args_pure),
    }


def place_value(val, sample_args, context=None):
    del val
    if context is None:
        context = comp_module.Context()

    entropy, sample_args = sample_args.peel()
    integer = num_module.integer(entropy, signed=False, min_abs=1)
    (entity,) = context.sample(sample_args, [integer])

    integer_as_str = str(integer)
    num_digits = len(integer_as_str)

    firsts = ['', 'ten ', 'hundred ']
    seconds = [
        'thousands', 'millions', 'billions', 'trillions', 'quadrillions',
        'quintillions', 'sextillions', 'septillions', 'octillions', 'nonillions',
        'decillions',
    ]
    place_names = ['units', 'tens', 'hundreds']
    for second in seconds:
        for first in firsts:
            place_names.append(first + second)

    place = random.randint(1, num_digits)
    place_name = place_names[place - 1]
    answer = sp_module.Integer(integer_as_str[num_digits - place])

    return ex_module.Problem(
        question=ex_module.question(
            context,
            'What is the {place_name} digit of {integer}?',
            place_name=place_name, integer=entity.expression_else_handle),
        answer=answer)


def round_number(val, sample_args, context=None):
    del val
    if context is None:
        context = comp_module.Context()

    entropy, sample_args = sample_args.peel()

    power = random.randint(-7, 6)
    answer_entropy = 1 + random.uniform(0, entropy / 2)
    entropy = max(1, entropy - answer_entropy)
    val_int = num_module.integer(answer_entropy, signed=True)

    remainder_divisor = 10 ** int(math.ceil(entropy))
    remainder_range_lower = -remainder_divisor / 2
    remainder_range_upper = remainder_divisor / 2

    if val_int <= 0:
        remainder_range_lower += 1
    if val_int >= 0:
        remainder_range_upper -= 1

    remainder = random.randint(remainder_range_lower, remainder_range_upper)
    input_ = val_int + sp_module.Rational(remainder, remainder_divisor)
    scale = 10**power if power >= 0 else sp_module.Rational(1, 10**(-power))
    input_ = input_ * scale
    value = val_int * scale
    if not num_module.is_integer(input_):
        input_ = disp_module.Decimal(input_)
    if not num_module.is_integer(value):
        value = disp_module.Decimal(value)

    (input_,) = context.sample(sample_args, [input_])

    if power > 0:
        round_to = 10**power
        if random.choice([False, True]):
            round_to = disp_module.StringNumber(round_to,
                                                join_number_words_with_hyphens=False)
        description = 'the nearest {round_to}'.format(round_to=round_to)
    elif power == 0 and random.choice([False, True]):
        description = 'the nearest integer'
    else:
        description = random.choice(['{dps} decimal place', '{dps} dp'])
        if power != -1:
            description += 's'
        dps = -power
        if random.choice([False, True]):
            dps = disp_module.StringNumber(dps)
        description = description.format(dps=dps)

    template = random.choice([
        'Round {input} to {description}.',
        'What is {input} rounded to {description}?',
    ])

    return ex_module.Problem(
        question=ex_module.question(
            context, template, input=input_, description=description),
        answer=value)


def _semi_prime(entropy):
    entropy += math.log10(max(1, entropy * math.log(10)))
    entropy_1, entropy_2 = entropy * np.random.dirichlet([1, 1])

    approx_1 = num_module.integer(entropy_1, signed=False, min_abs=2)
    approx_2 = num_module.integer(entropy_2, signed=False, min_abs=2)

    factor_1 = sp_module.ntheory.generate.randprime(approx_1 / 2, approx_1 * 2)
    factor_2 = sp_module.ntheory.generate.randprime(approx_2 / 2, approx_2 * 2)

    return factor_1 * factor_2


def is_prime(val, sample_args, context=None):
    del val
    if context is None:
        context = comp_module.Context()

    entropy, sample_args = sample_args.peel()
    composite = _semi_prime(entropy)

    if random.choice([False, True]):
        integer = composite
        is_prime_ = False
    else:
        integer = sp_module.ntheory.generate.nextprime(composite - 4)
        is_prime_ = True

    (integer_entity,) = context.sample(sample_args, [integer])

    if random.choice([False, True]) and integer != 1:
        answer = not is_prime_
        attr_name = random.choice(['composite', 'a composite number'])
    else:
        answer = is_prime_
        attr_name = random.choice(['prime', 'a prime number'])

    return ex_module.Problem(
        question=ex_module.question(
            context, 'Is {integer} {attr}?',
            integer=integer_entity.expression_else_handle,
            attr=attr_name),
        answer=answer)


def is_factor(val, sample_args, context=None):
    del val
    if context is None:
        context = comp_module.Context()

    entropy, sample_args = sample_args.peel()

    entropy_factor = 1 + random.uniform(0, entropy / 3)
    entropy = max(0, entropy - entropy_factor)
    maybe_factor = num_module.integer(entropy_factor, False, min_abs=2)

    integer = maybe_factor * num_module.integer(entropy, False, min_abs=1)

    if random.choice([False, True]):
        integer += random.randint(1, maybe_factor - 1)

    (entity,) = context.sample(sample_args, [integer])

    templates = [
        'Is {maybe_factor} a factor of {val}?',
        'Is {val} a multiple of {maybe_factor}?',
        'Does {maybe_factor} divide {val}?',
    ]
    if maybe_factor == 2:
        templates += [
            'Is {val} even?',
        ]
    template = random.choice(templates)

    answer = integer % maybe_factor == 0
    return ex_module.Problem(
        question=ex_module.question(
            context, template, maybe_factor=maybe_factor,
            val=entity.expression_else_handle),
        answer=answer)


def list_prime_factors(val, sample_args, context=None):
    del val
    if context is None:
        context = comp_module.Context()

    entropy, sample_args = sample_args.peel()
    entropy = max(1, entropy)

    integer = num_module.integer(entropy, signed=False, min_abs=2)

    (entity,) = context.sample(sample_args, [integer])
    prime_factors = sorted(sp_module.factorint(integer).keys())
    template = random.choice([
        'What are the prime factors of {val}?',
        'List the prime factors of {val}.',
    ])
    return ex_module.Problem(
        question=ex_module.question(
            context, template, val=entity.expression_else_handle),
        answer=disp_module.NumberList(prime_factors))


def _pair_with_large_hidden_factor(entropy):
    entropy_p, entropy_q, _ = entropy * np.random.dirichlet([1, 1, 1])
    entropy_p = max(1, entropy_p)
    entropy_q = max(1, entropy_q)
    entropy_mult = max(0, entropy - entropy_p - entropy_q)

    p = num_module.integer(entropy_p, False, min_abs=1)
    q = num_module.integer(entropy_q, False, min_abs=1)
    mult = num_module.integer(entropy_mult, False, min_abs=1)
    p *= mult
    q *= mult
    return p, q


def lcm(val, sample_args, context=None):
    del val
    if context is None:
        context = comp_module.Context()

    entropy, sample_args = sample_args.peel()

    p, q = _pair_with_large_hidden_factor(entropy)
    answer = sp_module.lcm(p, q)

    if random.choice([False, True]):
        p, q = context.sample(sample_args, [p, q])
        adjective = random.choice(['least', 'lowest', 'smallest'])
        template = random.choice([
            'Calculate the {adj} common multiple of {p} and {q}.',
            'What is the {adj} common multiple of {p} and {q}?',
        ])
        return ex_module.Problem(
            question=ex_module.question(
                context, template, adj=adjective,
                p=p.expression_else_handle,
                q=q.expression_else_handle),
            answer=answer)
    else:
        p = num_module.integer(2, signed=True, coprime_to=p) / p
        q = num_module.integer(2, signed=True, coprime_to=q) / q
        p, q = context.sample(sample_args, [p, q])

        template = random.choice([
            'What is the common denominator of {p} and {q}?',
            'Find the common denominator of {p} and {q}.',
            'Calculate the common denominator of {p} and {q}.',
        ])
        return ex_module.Problem(
            question=ex_module.question(
                context, template, p=p.expression_else_handle,
                q=q.expression_else_handle),
            answer=answer)


def _random_coprime_pair(entropy):
    coprime_product = num_module.integer(entropy, False, min_abs=1)
    factors = sp_module.factorint(coprime_product)

    def take():
        prime = random.choice(list(factors.keys()))
        power = factors[prime]
        del factors[prime]
        return prime ** power

    if random.random() < 0.8 and len(factors) >= 2:
        count_left = random.randint(1, len(factors) - 1)
        count_right = len(factors) - count_left
    else:
        count_left = random.randint(0, len(factors))
        count_right = len(factors) - count_left

    left = sp_module.prod([take() for _ in range(count_left)])
    right = sp_module.prod([take() for _ in range(count_right)])
    assert left * right == coprime_product
    return left, right


def gcd(val, sample_args, context=None):
    del val
    if context is None:
        context = comp_module.Context()

    entropy, sample_args = sample_args.peel()

    if val is None:
        value_entropy = 1 + random.uniform(0, entropy / 3)
        entropy = max(1, entropy - value_entropy)
        val = num_module.integer(value_entropy, False, min_abs=1)

    p_mult, q_mult = _random_coprime_pair(entropy)

    p = val * p_mult
    q = val * q_mult
    assert sp_module.gcd(p, q) == val

    p, q = context.sample(sample_args, [p, q])

    adj = (random.choice(['greatest', 'highest']) + ' common '
           + random.choice(['divisor', 'factor']))

    return ex_module.Problem(
        question=ex_module.question(
            context, 'Calculate the {adj} of {p} and {q}.',
            adj=adj, p=p, q=q),
        answer=val)


def div_remainder(val, sample_args, context=None):
    del val
    if context is None:
        context = comp_module.Context()

    entropy, sample_args = sample_args.peel()

    if val is None:
        entropy_value = 1 + random.uniform(0, entropy / 3)
        entropy = max(0, entropy - entropy_value)
        val = num_module.integer(entropy_value, signed=False)

    entropy_a, entropy_q = entropy * np.random.dirichlet([1, 1])
    a = num_module.integer(entropy_a, signed=False, min_abs=1)
    q = val + num_module.integer(entropy_q, signed=False, min_abs=1)

    p = a * q + val
    assert p % q == val
    p, q = context.sample(sample_args, [p, q])

    template = random.choice([
        'Calculate the remainder when {p} is divided by {q}.',
        'What is the remainder when {p} is divided by {q}?',
    ])
    return ex_module.Problem(
        question=ex_module.question(
            context, template, p=p.expression_else_handle,
            q=q.expression_else_handle),
        answer=val)


def base_conversion(min_entropy, max_entropy):
    context = comp_module.Context()

    from_base = random.randint(2, 16)
    while True:
        to_base = random.randint(2, 16)
        if to_base != from_base:
            break

    entropy_used = math.log10(16 * 15)
    entropy = random.uniform(
        min_entropy - entropy_used, max_entropy - entropy_used)

    val = num_module.integer(entropy, signed=True)
    template = random.choice([
        '{from_str} (base {from_base}) to base {to_base}',
        'Convert {from_str} (base {from_base}) to base {to_base}.',
        'What is {from_str} (base {from_base}) in base {to_base}?',
    ])
    return ex_module.Problem(
        question=ex_module.question(
            context, template,
            from_str=disp_module.NumberInBase(val, from_base),
            from_base=from_base,
            to_base=to_base),
        answer=disp_module.NumberInBase(val, to_base))
