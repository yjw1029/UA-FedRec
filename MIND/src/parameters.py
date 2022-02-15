import os
import sys
import logging
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.getenv("AMLT_DATA_DIR", "../data"),
        help="path to downloaded raw adressa dataset",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.getenv("AMLT_OUTPUT_DIR", "../output"),
        help="path to downloaded raw adressa dataset",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mind",
        choices=["mind", "adressa", "feeds"],
        help="decide which dataset for preprocess",
    )

    # job configutation
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--job_name", type=str, default="FedRec", help="name to choose job config including task, dataset, model etc..")
    parser.add_argument("--project_name", type=str, default="fl-attack", help="Wandb project name.")
    parser.add_argument("--run_name", type=str, default="", help="Wandb run name.")

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--user_num", type=int, default=50)
    parser.add_argument("--max_his_len", type=float, default=50)
    parser.add_argument(
        "--npratio",
        type=int,
        default=20,
        help="randomly sample npratio negative behaviors for every positive behavior",
    )
    parser.add_argument("--freeze_embedding", type=str2bool, default=False, help="Whether to freeze word embedding in model.")
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="0")

    # for attack
    parser.add_argument("--mal_user_ratio", type=float, default=0.01)
    # model poison attack
    parser.add_argument("--mal_factor", type=float, default=3.0, help="ambiguation degree.")
    # Attack-FedRec
    parser.add_argument("--mal_factor1", type=float, default=3.0, help="ambiguation degree for news similarity perturabtion.")
    parser.add_argument("--mal_factor2", type=float, default=3.0, help="ambiguation degree for user model perturbation.")
    parser.add_argument("--mal_factor3", type=float, default=3.0, help="ambiguation degree for quantity perturbation.")
    # mal adv item
    parser.add_argument("--mal_adv_lr", type=float, default=1.0, help="Adverserial attack learning rate")
    # mal adv round
    parser.add_argument("--adv_update_round", type=int, default=100, help="Round to update nearear neighbor")
    parser.add_argument("--adv_alpha", type=float, default=1.0, help="Alpha in adverserial recommendation")


    # for robust aggregation
    # trimmed_mean
    parser.add_argument("--trimmed_mean_beta", type=int, default=1, help="Number of largest or smaller gradient to remove.")
    # norm bounding
    parser.add_argument("--norm_bound", type=float, default=10.0, help="Norm bound for robust norm bounding aggregation.")
    # krum
    parser.add_argument("--krum_mal_num", type=int, default=1, help="Number of malicious users estimated in krum.")
    # multi-krum
    parser.add_argument("--multi_krum_num", type=int, default=49, help="Number of user selected by mutli-krum")
    
    args = parser.parse_args()
    return args


def setuplogger(args, out_path):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    fh = logging.FileHandler(out_path / f"log.{args.mode}.txt")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)