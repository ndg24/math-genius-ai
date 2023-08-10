from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hashlib


def is_train(number):
  value_as_string = str(number).encode('utf-8')
  return int(hashlib.md5(value_as_string).hexdigest(), 16) % 2 == 0
