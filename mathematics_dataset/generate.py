from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import textwrap

from absl import app as absl_app
from absl import flags as absl_flags
from absl import logging as absl_logging
from mathematics_dataset import generate_settings_math
from mathematics_dataset.modules import modules_math
import six
from six.moves import range as six_range


FLAGS_MATH = absl_flags.FLAGS

absl_flags.DEFINE_string('math_filter', 'linear_1d', 'restrict to matching module names')
absl_flags.DEFINE_integer('math_per_train_module', 10000, 'Num of examples per train module')
absl_flags.DEFINE_integer('math_per_test_module', 10000, 'Num of examples per test module')
absl_flags.DEFINE_bool('math_show_dropped', False, 'Whether to print dropped questions')


filtered_modules_math = collections.OrderedDict([])
counts_math = {}


def _make_entropy_fn_math(level, num_levels):
    lower = level / num_levels
    upper = (level + 1) / num_levels
    def modify_entropy_math(range_):
        assert len(range_) == 2
        length = range_[1] - range_[0]
        return (range_[0] + lower * length, range_[0] + upper * length)
    return modify_entropy_math


def _filter_and_flatten_math(modules_):
    """Returns flattened dict, filtered according to FLAGS_MATH."""
    flat = collections.OrderedDict()

    def add(submodules, prefix=None):
        for key, module_or_function in six.iteritems(submodules):
            full_name = prefix + '__' + key if prefix is not None else key
            if isinstance(module_or_function, dict):
                add(module_or_function, full_name)
            else:
                if FLAGS_MATH.math_filter not in full_name:
                    continue
                flat[full_name] = module_or_function

    add(modules_)

    # Make sure list of modules are in deterministic order. This is important when
    # generating across multiple machines.
    flat = collections.OrderedDict(
        [(key, flat[key]) for key in sorted(six.iterkeys(flat))])

    return flat


def init_modules_math(train_split=False):
    """Inits the dicts containing functions for generating modules."""
    if filtered_modules_math:
        return  # already initialized

    all_modules_math = collections.OrderedDict([])
    if train_split:
        all_modules_math['train-easy'] = modules_math.train(_make_entropy_fn_math(0, 3))
        all_modules_math['train-medium'] = modules_math.train(_make_entropy_fn_math(1, 3))
        all_modules_math['train-hard'] = modules_math.train(_make_entropy_fn_math(2, 3))
    else:
        all_modules_math['train'] = modules_math.train(_make_entropy_fn_math(0, 1))

    all_modules_math['interpolate'] = modules_math.test()
    all_modules_math['extrapolate'] = modules_math.test_extra()

    counts_math['train'] = FLAGS_MATH.math_per_train_module
    counts_math['train-easy'] = FLAGS_MATH.math_per_train_module // 3
    counts_math['train-medium'] = FLAGS_MATH.math_per_train_module // 3
    counts_math['train-hard'] = FLAGS_MATH.math_per_train_module // 3
    counts_math['interpolate'] = FLAGS_MATH.math_per_test_module
    counts_math['extrapolate'] = FLAGS_MATH.math_per_test_module

    for regime_, modules_ in six.iteritems(all_modules_math):
        filtered_modules_math[regime_] = _filter_and_flatten_math(modules_)


def sample_from_module_math(module):
    num_dropped = 0
    while True:
        problem = module()
        question = str(problem.question)
        if len(question) > generate_settings_math.MAX_QUESTION_LENGTH:
            num_dropped += 1
            if FLAGS_MATH.math_show_dropped:
                absl_logging.warning('Dropping question: %s', question)
            continue
        answer = str(problem.answer)
        if len(answer) > generate_settings_math.MAX_ANSWER_LENGTH:
            num_dropped += 1
            if FLAGS_MATH.math_show_dropped:
                absl_logging.warning('Dropping question with answer: %s', answer)
            continue
        return problem, num_dropped


def main_math(unused_argv):
    init_modules_math()

    text_wrapper = textwrap.TextWrapper(
        width=80, initial_indent=' ', subsequent_indent='  ')

    for regime, flat_modules in six.iteritems(filtered_modules_math):
        per_module = counts_math[regime]
        for module_name, module in six.iteritems(flat_modules):
            # These magic print constants make the header bold.
            print('\033[1m{}/{}\033[0m'.format(regime, module_name))
            num_dropped = 0
            for _ in six_range(per_module):
                problem, extra_dropped = sample_from_module_math(module)
                num_dropped += extra_dropped
                text = text_wrapper.fill(
                    '{}  \033[92m{}\033[0m'.format(problem.question, problem.answer))
                print(text)
            if num_dropped > 0:
                absl_logging.warning('Dropped %d examples', num_dropped)


if __name__ == '__main__':
    absl_app.run(main_math)