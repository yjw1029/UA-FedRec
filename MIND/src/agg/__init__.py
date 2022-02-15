from agg.base import BaseAggregator
from agg.user import UserAggregator

from agg.robust import *

def get_agg(name):
    return eval(name)