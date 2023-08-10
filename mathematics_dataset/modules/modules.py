from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mathematics_dataset.modules import algebra as a_module
from mathematics_dataset.modules import arithmetic as ar_module
from mathematics_dataset.modules import calculus as c_module
from mathematics_dataset.modules import comparison as co_module
from mathematics_dataset.modules import measurement as m_module
from mathematics_dataset.modules import numbers as n_module
from mathematics_dataset.modules import polynomials as p_module
from mathematics_dataset.modules import probability as pr_module
import six


all_ = {
    'algebra': a_module,
    'arithmetic': ar_module,
    'calculus': c_module,
    'comparison': co_module,
    'measurement': m_module,
    'numbers': n_module,
    'polynomials': p_module,
    'probability': pr_module,
}


def train(entropy_fn):
    """Returns dictionary of training modules."""
    return {
        module_name: module.train(entropy_fn)
        for module_name, module in six.iteritems(all_)
    }


def test():
    """Returns dictionary of testing modules."""
    return {
        module_name: module.test()
        for module_name, module in six.iteritems(all_)
    }


def test_extra():
    """Returns dictionary of extrapolation testing modules."""
    return {
        module_name: module.test_extra()
        for module_name, module in six.iteritems(all_)
    }
