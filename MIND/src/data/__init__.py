from data.test import NewsDataset, UserDataset

from data.base import TrainBaseDataset, train_base_collate_fn
from data.ls import TrainLSDataset, train_ls_collate_fn
from data.pp import TrainPPDataset
from data.mp import TrainMPDataset, train_mp_collate_fn

def get_dataset(name):
    return eval(name)

def get_collate_fn(name):
    if name is None:
        return None
    return eval(name)