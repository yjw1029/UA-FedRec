from transformers import BertTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import pickle
import argparse
import re
from collections import Counter
# config

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
        help="path to save processed dataset, default in ../data",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mind",
        choices=["mind", "adressa", "feeds"],
        help="decide which dataset for preprocess"
    )
    parser.add_argument(
        "--npratio",
        type=int,
        default=4
    )
    parser.add_argument(
        "--max_his_len", type=int, default=50
    )
    parser.add_argument("--min_word_cnt", type=int, default=3)
    parser.add_argument("--max_title_len", type=int, default=30)
    parser.add_argument("--glove_path", type=str, default="../raw/glove/glove.840B.300d.txt")

    args = parser.parse_args()
    return args


def word_tokenize(sent):
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

if __name__ == "__main__":
    args = parse_args()
    raw_path = Path(args.raw_path) / args.data
    out_path = Path(args.out_path) / args.data

    if not raw_path.is_dir():
        raise ValueError(f"{raw_path.name} does not exist.")

    out_path.mkdir(exist_ok=True, parents=True)

    # news preprocess
    nid2index = {"<unk>": 0}
    news_info = {"<unk>": ""}
    word_cnt = Counter()

    for l in tqdm(open(raw_path / "train" / "news.tsv", "r", encoding='utf-8')):
        nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
        if nid in nid2index:
            continue
        tokens = word_tokenize(title)[:args.max_title_len]
        nid2index[nid] = len(nid2index)
        news_info[nid] = tokens
        word_cnt.update(tokens)


    for l in tqdm(open(raw_path / "valid" / "news.tsv", "r", encoding='utf-8')):
        nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
        if nid in nid2index:
            continue
        tokens = word_tokenize(title)[:args.max_title_len]
        nid2index[nid] = len(nid2index)
        news_info[nid] = tokens
        word_cnt.update(tokens)

    with open(out_path / "glove_nid2index.pkl", "wb") as f:
        pickle.dump(nid2index, f)

    with open(out_path / "glove_news_info.pkl", "wb") as f:
        pickle.dump(news_info, f)

    if os.path.exists(raw_path / "test"):
        test_news_info = {"<unk>": ""}
        test_nid2index = {"<unk>": 0}

        for l in tqdm(open(raw_path / "test" / "news.tsv", "r", encoding='utf-8')):
            nid, vert, subvert, title, abst, url, ten, aen = l.strip("\n").split("\t")
            if nid in test_nid2index:
                continue
            tokens = word_tokenize(title)[:args.max_title_len]
            test_nid2index[nid] = len(test_nid2index)
            test_news_info[nid] = tokens

        with open(out_path / "glove_test_nid2index.pkl", "wb") as f:
            pickle.dump(test_nid2index, f)

        with open(out_path / "glove_test_news_info.pkl", "wb") as f:
            pickle.dump(test_news_info, f)

    vocab_dict = {"<unk>": 0}

    for w, c in tqdm(word_cnt.items()):
        if c >= args.min_word_cnt:
            vocab_dict[w] = len(vocab_dict)

    with open(out_path / "glove_vocab_dict.pkl", "wb") as f:
        pickle.dump(vocab_dict, f)

    news_index = np.zeros((len(news_info) + 1, args.max_title_len), dtype="float32")

    for nid in tqdm(nid2index):
        news_index[nid2index[nid]] = [
            vocab_dict[w] if w in vocab_dict else 0 for w in news_info[nid]
        ] + [0] * (args.max_title_len - len(news_info[nid]))

    np.save(out_path / "glove_news_index", news_index)

    if os.path.exists(raw_path / "test"):
        test_news_index = np.zeros((len(test_news_info) + 1, args.max_title_len), dtype="float32")

        for nid in tqdm(test_nid2index):
            test_news_index[test_nid2index[nid]] = [
                vocab_dict[w] if w in vocab_dict else 0 for w in test_news_info[nid]
            ] + [0] * (args.max_title_len - len(test_news_info[nid]))

        np.save(out_path / "glove_test_news_index", test_news_index)


    def load_matrix(glove_path, word_dict):
        # embebbed_dict = {}
        embedding_matrix = np.zeros((len(word_dict) + 1, 300))
        exist_word = []

        # get embedded_dict
        with open(glove_path, "rb") as f:
            for l in tqdm(f):
                l = l.split()
                word = l[0].decode()
                if len(word) != 0 and word in word_dict:
                    wordvec = [float(x) for x in l[1:]]
                    index = word_dict[word]
                    embedding_matrix[index] = np.array(wordvec)
                    exist_word.append(word)

        # get union
        return embedding_matrix, exist_word


    embedding_matrix, exist_word = load_matrix(args.glove_path, vocab_dict)

    print(embedding_matrix.shape[0], len(exist_word))

    np.save(out_path / "glove_embedding", embedding_matrix)

