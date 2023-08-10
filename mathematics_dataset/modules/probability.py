import collections
import functools
import random
import string

from mathematics_dataset import example
from mathematics_dataset.modules import train_test_split
from mathematics_dataset.util import combinatorics
from mathematics_dataset.util import composition
from mathematics_dataset.util import display
from mathematics_dataset.util import probability
import numpy as np
from six.moves import range
from six.moves import zip

LETTERS = string.ascii_lowercase
MAX_FRAC_TRIVIAL_PROB = 0.1
MAX_DISTINCT_LETTERS = 6
MAX_TOTAL_LETTERS = 20
MAX_LETTER_REPEAT = 10
SWR_SAMPLE_COUNT = [2, 4]
SWR_SAMPLE_COUNT_EXTRAPOLATE = [5, 5]
GERUNDS = {'pick': 'picking'}

def make_modules(is_training):
    return {
        'swr_p_sequence': functools.partial(swr_prob_sequence, is_train=is_training, sample_range=SWR_SAMPLE_COUNT),
        'swr_p_level_set': functools.partial(swr_prob_level_set, is_train=is_training, sample_range=SWR_SAMPLE_COUNT),
    }

def train(entropy_fn):
    return make_modules(is_training=True)

def test():
    return make_modules(is_training=False)

def test_extra():
    return {
        'swr_p_sequence_more_samples': functools.partial(swr_prob_sequence, is_train=None, sample_range=SWR_SAMPLE_COUNT_EXTRAPOLATE),
        'swr_p_level_set_more_samples': functools.partial(swr_prob_level_set, is_train=None, sample_range=SWR_SAMPLE_COUNT_EXTRAPOLATE),
    }

LetterBag = collections.namedtuple('LetterBag', ('weights', 'random_variable', 'letters_distinct', 'bag_contents'))

def sample_letter_bag(is_train, min_total):
    while True:
        num_distinct_letters = random.randint(1, MAX_DISTINCT_LETTERS)
        num_letters_total = random.randint(max(num_distinct_letters, min_total), min(MAX_TOTAL_LETTERS, num_distinct_letters * MAX_LETTER_REPEAT))
        letter_counts = combinatorics.uniform_positive_integers_with_sum(num_distinct_letters, num_letters_total)
        if (is_train is None or train_test_split.is_train(sorted(letter_counts)) == is_train):
            break
    letters_distinct = random.sample(LETTERS, num_distinct_letters)
    weights = {i: 1 for i in range(num_letters_total)}
    letters_with_repetition = []
    for letter, count in zip(letters_distinct, letter_counts):
        letters_with_repetition += [letter] * count
    random.shuffle(letters_with_repetition)
    random_variable = probability.DiscreteRandomVariable({i: letter for i, letter in enumerate(letters_with_repetition)})
    if random.choice([False, True]):
        bag_contents = ''.join(letters_with_repetition)
    else:
        letters_and_counts = ['{}: {}'.format(letter, count) for letter, count in zip(letters_distinct, letter_counts)]
        bag_contents = '{' + ', '.join(letters_and_counts) + '}'
    return LetterBag(weights=weights, random_variable=random_variable, letters_distinct=letters_distinct, bag_contents=bag_contents)

def swr_space(is_train, sample_range):
    num_sampled = random.randint(*sample_range)
    sample = sample_letter_bag(is_train=is_train, min_total=num_sampled)
    space = probability.SampleWithoutReplacementSpace(sample.weights, num_sampled)
    random_variable = probability.FiniteProductRandomVariable([sample.random_variable] * num_sampled)
    random_variable.description = str(display.StringNumber(num_sampled)) + ' letters picked without replacement from ' + sample.bag_contents
    return sample.letters_distinct, space, random_variable

def sample_without_replacement_probability_question(is_train, event_fn, sample_range):
    def too_big(event_in_space):
        if isinstance(event_in_space, probability.SequenceEvent):
            size = len(event_in_space.all_sequences())
        else:
            assert isinstance(event_in_space, probability.FiniteProductEvent)
            size = np.prod([len(event.values) for event in event_in_space.events])
        return size > int(2e5)
    allow_trivial_prob = random.random() < MAX_FRAC_TRIVIAL_PROB
    while True:
        distinct_letters, space, random_variable = swr_space(is_train, sample_range)
        event, event_description = event_fn(values=distinct_letters, length=space.n_samples, verb='pick')
        event_in_space = random_variable.inverse(event)
        if too_big(event_in_space):
            continue
        answer = space.probability(event_in_space)
        if answer not in [0, 1] or allow_trivial_prob:
            break
    context = composition.Context()
    template = random.choice([
        '{random_variable_capitalize}. What is prob of {event}?',
        '{random_variable_capitalize}. Give prob of {event}.',
        'What is prob of {event} when {random_variable}?',
        'Calculate prob of {event} when {random_variable}.',
    ])
    question = example.question(
        context,
        template,
        random_variable=random_variable.description,
        random_variable_capitalize=str(random_variable.description).capitalize(),
        event=event_description)
    return example.Problem(question, answer)

def swr_prob_sequence(is_train, sample_range):
    return sample_without_replacement_probability_question(is_train=is_train, event_fn=_sequence_event, sample_range=sample_range)

def swr_prob_level_set(is_train, sample_range):
    return sample_without_replacement_probability_question(is_train=is_train, event_fn=_level_set_event, sample_range=sample_range)
