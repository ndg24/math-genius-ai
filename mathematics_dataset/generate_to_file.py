from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl import app as my_app
from absl import flags as my_flags
from absl import logging as my_logging
from mathematics_dataset import generate_math
import six
from six.moves import range as six_range

FLAGS_MATH = my_flags.FLAGS

my_flags.DEFINE_string('output_directory', None, 'Where to write output text')
my_flags.DEFINE_boolean('is_train_split', True,
                        'Whether to split training data by difficulty')
my_flags.mark_flag_as_required('output_directory')


def main_math(unused_argv):
    generate_math.init_modules_math(FLAGS_MATH.is_train_split)

    output_dir = os.path.expanduser(FLAGS_MATH.output_directory)
    if os.path.exists(output_dir):
        my_logging.fatal('output directory %s already exists', output_dir)
    my_logging.info('Writing to %s', output_dir)
    os.makedirs(output_dir)

    for regime, flat_modules in six.iteritems(generate_math.filtered_modules_math):
        regime_dir = os.path.join(output_dir, regime)
        os.mkdir(regime_dir)
        per_module = generate_math.counts_math[regime]
        for module_name, module in six.iteritems(flat_modules):
            path = os.path.join(regime_dir, module_name + '.txt')
            with open(path, 'w') as text_file:
                for _ in six_range(per_module):
                    problem, _ = generate_math.sample_from_module_math(module)
                    text_file.write(str(problem.question) + '\n')
                    text_file.write(str(problem.answer) + '\n')
            my_logging.info('Written %s', path)


if __name__ == '__main__':
    my_app.run(main_math)