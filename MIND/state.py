from pathlib import Path
from collections import defaultdict
import numpy as np

data_dir = "./raw/mind/train/behaviors.tsv"

user_samples = defaultdict(int)

with open(data_dir, 'r') as f:
    for l in f:
        imp_id, uid, t, his, imprs = l.strip("\n").split("\t")
        user_samples[uid] += 1

mean = np.mean(list(user_samples.values()))
std = np.std(list(user_samples.values()))
max = np.max(list(user_samples.values()))

print(mean, std, max)