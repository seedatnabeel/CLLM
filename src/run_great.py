import pandas as pd
import torch
from src.models import *
from src.utils import *
from src.data_loader import *
from copy import deepcopy

# Split remaining data into validation and test sets
from sklearn.model_selection import train_test_split
from copy import deepcopy
import pickle
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run baselines on a dataset.")
    parser.add_argument(
        "--dataset", type=str, default="seer", help="Name of the dataset"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--ns", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    ns = args.ns
    dataset = args.dataset
    seed = args.seed
    n_synthetic = 1000

    print(f"Running {dataset} {seed} {ns}")

    # covid, seer, cutract, support, maggic, adult, compas [drug, fraud]

    df_feat, df_label, df = get_data(dataset=dataset, seed=seed)

    X_train, X_remain, y_train, y_remain = sample_and_split(
        df_feat, df_label, ns=ns, seed=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_remain, y_remain, test_size=0.5, random_state=seed
    )

    X_train_orig = deepcopy(X_train)
    y_train_orig = deepcopy(y_train)

    results = {}

    results["Original"] = {"X": X_train_orig, "y": y_train_orig}
    results["Oracle"] = {"X": X_val, "y": y_val}
    results["Test"] = {"X": X_test, "y": y_test}

    # # apply Great
    X_syn, y_syn = great(
        X_train_orig,
        y_train_orig,
        n_synthetic=n_synthetic,
        epochs=1000,
        max_length=4000,
    )

    results["great"] = {"X": X_syn, "y": y_syn}

    with open(f"save_dfs/great_pipeline_{dataset}_{seed}_{ns}.pickle", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
