from model.nrms import *
from model.lstur import *

def get_model(name):
    return eval(name)