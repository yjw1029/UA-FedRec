from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import os
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        type=str,
        default="../raw/",
        help="path to raw mind dataset or parsed ",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="../data/",
        help="path to save processed dataset, default in ../raw/mind/preprocess",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mind",
        choices=["mind", "adressa", "feeds"],
        help="decide which dataset for preprocess",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    raw_path = Path(args.raw_path) / args.data
    out_path = Path(args.out_path) / args.data

    if not raw_path.is_dir():
        raise ValueError(f"{raw_path.name} does not exist.")

    out_path.mkdir(exist_ok=True, parents=True)

    pos_cnt = Counter()
    neg_cnt = Counter()

    # read user impressions
    for l in tqdm(open(raw_path / "train" / "behaviors.tsv", "r")):
        imp_id, uid, _, _, imprs = l.strip("\n").split("\t")
        imprs = [i.split("-") for i in imprs.split(" ")]
        neg_imp = [i[0] for i in imprs if i[1] == "0"]
        pos_imp = [i[0] for i in imprs if i[1] == "1"]

        pos_cnt.update(pos_imp)
        neg_cnt.update(neg_imp)

    news_ctr = {}
    for n in pos_cnt:
        overall_cnt = pos_cnt[n] + neg_cnt[n]
        news_ctr[n] = pos_cnt[n] / overall_cnt

    with open(out_path / "news_ctr.pkl", "wb") as f:
        pickle.dump(news_ctr, f)

    news_ctr_sort = sorted(news_ctr.items(), key=lambda x: x[1], reverse=True)
    all_news_num = len(news_ctr_sort)

    popular_news_num = int(all_news_num * 0.1)
    low_pop_news_num = int(all_news_num * 0.55)
    medium_news_num = all_news_num - popular_news_num - low_pop_news_num

    pop_news = news_ctr_sort[:popular_news_num]
    medium_news = news_ctr_sort[popular_news_num : popular_news_num + medium_news_num]
    low_pop_news = news_ctr_sort[popular_news_num + medium_news_num :]

    news_pop_class = {}
    for n, ctr in pop_news:
        news_pop_class[n] = 0

    for n, ctr in medium_news:
        news_pop_class[n] = 1

    for n, ctr in low_pop_news:
        news_pop_class[n] = 2

    with open(out_path / "news_pop_class.pkl", "wb") as f:
        pickle.dump(news_pop_class, f)
