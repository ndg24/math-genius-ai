from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from mathematics_dataset.util import composition_math


def generate_question_math(context, template, **kwargs):
    assert isinstance(context, composition_math.Context)
    assert isinstance(template, str)
    prefix, kwargs = composition_math.expand_entities(context, **kwargs)
    if prefix:
        prefix += ' '
    return prefix + template.format(**kwargs)


MathProblem = collections.namedtuple('MathProblem', ('question', 'answer'))