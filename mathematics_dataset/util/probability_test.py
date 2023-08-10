from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import itertools
import six
from six.moves import zip
import sympy

class Event(object):
    pass

class ProbabilitySpace(object):

    @abc.abstractmethod
    def probability(self, event):
        pass

class RandomVariable(object):

    @abc.abstractmethod
    def __call__(self, event):
        pass

    @abc.abstractmethod
    def inverse(self, event):
        pass

class DiscreteEvent(Event):

    def __init__(self, values):
        self._values = values

    @property
    def values(self):
        return self._values

class FiniteProductEvent(Event):

    def __init__(self, events):
        self._events = events

    @property
    def events(self):
        return self._events

    def all_sequences(self):
        if not all(isinstance(event, DiscreteEvent) for event in self._events):
            raise ValueError('Not all component events are DiscreteEvents')
        values_list = [event.values for event in self._events]
        return itertools.product(*values_list)

class CountLevelSetEvent(Event):

    def __init__(self, counts):
        self._counts = counts
        self._all_sequences = None

    @property
    def counts(self):
        return self._counts

    def all_sequences(self):
        if self._all_sequences is None:
            cache = {}
            labels = list(self._counts.keys())

            def generate(counts):
                if sum(counts) == 0:
                    return [()]
                counts = tuple(counts)
                if counts in cache:
                    return cache[counts]
                generated = []
                for i, count in enumerate(counts):
                    if count == 0:
                        continue
                    counts_minus = list(counts)
                    counts_minus[i] -= 1
                    counts_minus = tuple(counts_minus)
                    extensions = generate(counts_minus)
                    generated += [tuple([labels[i]] + list(extension))
                                  for extension in extensions]
                cache[counts] = generated
                return generated

            self._all_sequences = generate(list(self._counts.values()))

        return self._all_sequences

class SequenceEvent(Event):

    def __init__(self, sequences):
        self._sequences = sequences

    def all_sequences(self):
        return self._sequences

def normalize_weights(weights):
    weight_sum = sum(six.itervalues(weights))
    return {
        i: sympy.Rational(weight, weight_sum)
        for i, weight in six.iteritems(weights)
    }

class DiscreteProbabilitySpace(ProbabilitySpace):

    def __init__(self, weights=None):
        self._weights = normalize_weights(weights)

    def probability(self, event):
        if isinstance(event, DiscreteEvent):
            return sum(self._weights[value]
                       for value in event.values if value in self._weights)
        else:
            raise ValueError('Unhandled event type {}'.format(type(event)))

    @property
    def weights(self):
        return self._weights

class FiniteProductSpace(ProbabilitySpace):

    def __init__(self, spaces):
        self._spaces = spaces

    def all_spaces_equal(self):
        return all([self._spaces[0] == space for space in self._spaces])

    def probability(self, event):
        if isinstance(event, FiniteProductEvent):
            assert len(self._spaces) == len(event.events)
            return sympy.prod([
                space.probability(event_slice)
                for space, event_slice in zip(self._spaces, event.events)])

        if isinstance(event, CountLevelSetEvent) and self.all_spaces_equal():
            space = self._spaces[0]
            counts = event.counts
            probabilities = {
                value: space.probability(DiscreteEvent({value}))
                for value in six.iterkeys(counts)
            }

            num_events = sum(six.itervalues(counts))
            assert num_events == len(self._spaces)
            coeff = (
                sympy.factorial(num_events) / sympy.prod(
                    [sympy.factorial(i) for i in six.itervalues(counts)]))
            return coeff * sympy.prod([
                pow(probabilities[value], counts[value])
                for value in six.iterkeys(counts)
            ])

        raise ValueError('Unhandled event type {}'.format(type(event)))

    @property
    def spaces(self):
        return self._spaces

class SampleWithoutReplacementSpace(ProbabilitySpace):

    def __init__(self, weights, n_samples):
        if n_samples > len(weights):
            raise ValueError('n_samples is more than number of discrete elements')
        self._weights = normalize_weights(weights)
        self._n_samples = n_samples

    @property
    def n_samples(self):
        return self._n_samples

    def probability(self, event):
        try:
            all_sequences = event.all_sequences()
        except AttributeError:
            raise ValueError('Unhandled event type {}'.format(type(event)))

        probability_sum = 0
        for sequence in all_sequences:
            if len(sequence) != len(set(sequence)):
                continue
            p_sequence = 1
            removed_prob = 0
            for i in sequence:
                p = self._weights[i] if i in self._weights else 0
                if p == 0:
                    p_sequence = 0
                    break
                p_sequence *= p / (1 - removed_prob)
                removed_prob += p
            probability_sum += p_sequence
        return probability_sum

class IdentityRandomVariable(RandomVariable):

    def __call__(self, event):
        return event

    def inverse(self, event):
        return event

class DiscreteRandomVariable(RandomVariable):

    def __init__(self, mapping):
        self._mapping = mapping
        self._inverse = {}
        for key, value in six.iteritems(mapping):
            if value in self._inverse:
                self._inverse[value].add(key)
            else:
                self._inverse[value] = set([key])

    def __call__(self, event):
        if isinstance(event, DiscreteEvent):
            return DiscreteEvent({self._mapping[value] for value in event.values})
        else:
            raise ValueError('Unhandled event type {}'.format(type(event)))

    def inverse(self, event):
        if isinstance(event, DiscreteEvent):
            set_ = set()
            for value in event.values:
                if value in self._inverse:
                    set_.update(self._inverse[value])
            return DiscreteEvent(set_)
        else:
            raise ValueError('Unhandled event type {}'.format(type(event)))

class FiniteProductRandomVariable(RandomVariable):

    def __init__(self, random_variables):
        self._random_variables = random_variables

    def __call__(self, event):
        if isinstance(event, FiniteProductEvent):
            assert len(event.events) == len(self._random_variables)
            zipped = list(zip(self._random_variables, event.events))
            return FiniteProductEvent(
                [random_variable(sub_event)
                 for random_variable, sub_event in zipped])
        else:
            raise ValueError('Unhandled event type {}'.format(type(event)))

    def inverse(self, event):
        if isinstance(event, FiniteProductEvent):
            assert len(event.events) == len(self._random_variables)
            zipped = list(zip(self._random_variables, event.events))
            return FiniteProductEvent(tuple(
                random_variable.inverse(sub_event)
                for random_variable, sub_event in zipped))

        try:
            all_sequences = event.all_sequences()
        except AttributeError:
            raise ValueError('Unhandled event type {}'.format(type(event)))

        mapped = set()
        for sequence in all_sequences:
            assert len(sequence) == len(self._random_variables)
            zipped = list(zip(self._random_variables, sequence))
            mapped_sequence = FiniteProductEvent(tuple(
                random_variable.inverse(DiscreteEvent({element}))
                for random_variable, element in zipped))
            mapped.update(mapped_sequence.all_sequences())
        return SequenceEvent(mapped)
