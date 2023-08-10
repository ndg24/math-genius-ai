from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from absl import logging
from mathematics_dataset.sample import number as sample_number
from mathematics_dataset.util import display
import numpy as np
import six
from six.moves import zip
import sympy

MUL_SYM = '*'
DIV_SYM = '/'
POW_SYM = '**'
GT_SYM = '>'
LT_SYM = '<'
GE_SYM = '>='
LE_SYM = '<='
EQ_SYM = '='
NE_SYM = '!='

_EQ_PREC = 0
_CONSTANT_PREC = 1
_POW_PREC = 2
_SQRT_PREC = 3
_MUL_PREC = 4
_ADD_PREC = 5

def bracketed(node, parent, need_brackets):
    if not isinstance(node, Op):
        node = ConstantOp(node)
    node_prec = node.precedence
    parent_prec = parent.precedence
    if (parent_prec > node_prec or 
        (parent_prec == node_prec and not need_brackets)):
        return str(node)
    else:
        return f'({node})'

def _flatten(iterable):
    if isinstance(iterable, (list, tuple)):
        result = list(iterable)
    else:
        assert isinstance(iterable, dict)
        keys = sorted(six.iterkeys(iterable))
        result = [iterable[key] for key in keys]
    for item in result:
        assert not isinstance(item, (list, tuple, dict))
    return result

def _pack_sequence_as(example, flat):
    if isinstance(example, list) or isinstance(example, tuple):
        return flat
    else:
        assert isinstance(example, dict)
        keys = sorted(six.iterkeys(example))
        return {key: value for key, value in zip(keys, flat)}

@six.add_metaclass(abc.ABCMeta)
class Op(object):
    def __init__(self, children):
        assert isinstance(children, (list, dict, tuple))
        flat_children = _flatten(children)
        flat_children = [child if isinstance(child, Op) else ConstantOp(child)
                         for child in flat_children]
        children = _pack_sequence_as(children, flat_children)
        self._children = children

    @property
    def children(self):
        return self._children

    def descendants(self):
        descendants = [self]
        flat_children = _flatten(self._children)
        for child in flat_children:
            descendants += child.descendants()
        return descendants

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def sympy(self):
        pass

    def _sympy_(self):
        try:
            return self.sympy()
        except AttributeError as e:
            logging.error('Attribute error while trying to sympify: %s', e)
            raise e

    @abc.abstractproperty
    def precedence(self):
        pass

class ConstantOp(Op):
    def __init__(self, value):
        super(ConstantOp, self).__init__([])
        if isinstance(value, six.integer_types):
            value = sympy.Integer(value)
        self._value = value

    def __str__(self):
        return str(self._value)

    def sympy(self):
        return self._value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def _is_simple(self):
        if isinstance(self._value, sympy.Symbol):
            return True
        elif isinstance(self._value, int) or isinstance(self._value, sympy.Integer):
            return self._value >= 0
        elif isinstance(self._value, display.Decimal) or isinstance(self._value, np.int64) or isinstance(self._value, np.int32):
            return self._value >= 0
        elif isinstance(self._value, sympy.Rational):
            return False
        elif isinstance(self._value, sympy.Function):
            return True
        else:
            raise ValueError('Unknown type {}'.format(type(self._value)))

    @property
    def precedence(self):
        if self._is_simple():
            return _CONSTANT_PREC
        else:
            return _MUL_PREC

class _SumLikeOp(Op):
    @abc.abstractmethod
    def expanded_signs_and_terms(self):
        pass

    def __str__(self):
        signs_and_terms = self.expanded_signs_and_terms()
        if not signs_and_terms:
            return '0'
        expression = ''
        for i, (sign, term) in enumerate(signs_and_terms):
            if i == 0:
                if sign:
                    expression = bracketed(term, self, True)
                else:
                    expression = '-' + bracketed(term, self, True)
            else:
                if sign:
                    expression += f' + {bracketed(term, self, True)}'
                else:
                    expression += f' - {bracketed(term, self, True)}'
        return expression

class IdentityOp(_SumLikeOp):
    def __init__(self, input_):
        super(IdentityOp, self).__init__({'input': input_})

    def expanded_signs_and_terms(self):
        if isinstance(self.children['input'], _SumLikeOp):
            return self.children['input'].expanded_signs_and_terms()
        else:
            return [(True, self.children['input'])]

    def __str__(self):
        return str(self.children['input'])

    def sympy(self):
        return self.children['input'].sympy()

    @property
    def precedence(self):
        return self.children['input'].precedence

class NegOp(_SumLikeOp):
    def __init__(self, arg):
        super(NegOp, self).__init__({'input': arg})

    def expanded_signs_and_terms(self):
        if isinstance(self.children['input'], _SumLikeOp):
            inner_signs_and_terms = self.children['input'].expanded_signs_and_terms()
            return [(not sign, term) for (sign, term) in inner_signs_and_terms]
        else:
            return [(False, self.children['input'])]

    def sympy(self):
        return -sympy.sympify(self.children['input'])

    @property
    def precedence(self):
        return _ADD_PREC

class AddOp(_SumLikeOp):
    def __init__(self, *args):
        super(AddOp, self).__init__(args)

    def expanded_signs_and_terms(self):
        expanded = []
        for arg in self.children:
            if isinstance(arg, _SumLikeOp):
                expanded += arg.expanded_signs_and_terms()
            else:
                expanded.append((True, arg))
        return expanded

    def sympy(self):
        return sympy.Add(*[sympy.sympify(arg) for arg in self.children])

    @property
    def precedence(self):
        return _ADD_PREC

class SubOp(Op):
    def __init__(self, left, right):
        super(SubOp, self).__init__({'left': left, 'right': right})

    def __str__(self):
        return (bracketed(self.children['left'], self, False) + 
                ' - ' + bracketed(self.children['right'], self, True))

    def sympy(self):
        return sympy.Add(self.children['left'], sympy.Mul(-1, self.children['right']))

    @property
    def precedence(self):
        return _ADD_PREC

class MulOp(Op):
    def __init__(self, *args):
        super(MulOp, self).__init__(args)

    def __str__(self):
        if not self.children:
            return '1'
        args = [bracketed(arg, self, False) for arg in self.children]
        return MUL_SYM.join(args)

    def sympy(self):
        return sympy.Mul(*[sympy.sympify(arg) for arg in self.children])

    @property
    def precedence(self):
        return _MUL_PREC

class DivOp(Op):
    def __init__(self, numer, denom):
        super(DivOp, self).__init__({'numer': numer, 'denom': denom})

    def __str__(self):
        return f'{bracketed(self.children["numer"], self, True)}{DIV_SYM}' + \
               f'{bracketed(self.children["denom"], self, True)}'

    def sympy(self):
        return sympy.Mul(self.children['numer'], sympy.Pow(self.children['denom'], -1))

    @property
    def precedence(self):
        return _MUL_PREC

class PowOp(Op):
    def __init__(self, a, b):
        super(PowOp, self).__init__({'a': a, 'b': b})

    def __str__(self):
        return f'{bracketed(self.children["a"], self, True)}{POW_SYM}' + \
               f'{bracketed(self.children["b"], self, True)}'

    def sympy(self):
        return sympy.Pow(sympy.sympify(self.children['a']), sympy.sympify(self.children['b']))

    @property
    def precedence(self):
        return _POW_PREC

class SqrtOp(Op):
    def __init__(self, a):
        super(SqrtOp, self).__init__({'a': a})

    def __str__(self):
        return f'sqrt({self.children["a"]})'

    def sympy(self):
        return sympy.sqrt(self.children['a'])

    @property
    def precedence(self):
        return _POW_PREC

class EqOp(Op):
    def __init__(self, left, right):
        super(EqOp, self).__init__({'left': left, 'right': right})

    def __str__(self):
        return f'{self.children["left"]} = {self.children["right"]}'

    def sympy(self):
        return sympy.Eq(self.children['left'], self.children['right'])

    @property
    def precedence(self):
        return _EQ_PREC

def num_constants(expressions):
    if isinstance(expressions, Op):
        expressions = [expressions]
    descendants = []
    for expression in expressions:
        descendants += expression.descendants()
    candidates = [op for op in descendants if isinstance(op, ConstantOp)]
    return [constant for constant in candidates
            if sample_number.is_integer_or_rational_or_decimal(constant.value)]
