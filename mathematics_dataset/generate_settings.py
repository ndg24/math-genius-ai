from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string

MAX_Q_LEN = 160
MAX_A_LEN = 30
Q_CHARS = (
    ['', ' '] + list(string.ascii_letters + string.digits + string.punctuation))
EMPTY_IDX = Q_CHARS.index('')
NUM_IDXS = len(Q_CHARS)
CHAR_TO_IDX = {char: index for index, char in enumerate(Q_CHARS)}
IDX_TO_CHAR = {index: char for index, char in enumerate(Q_CHARS)}