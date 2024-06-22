import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)


def split_data(df, target_column, n_samples_per_class, random_state=42):
    train_data = (
        df.groupby(target_column)
        .apply(lambda x: x.sample(n_samples_per_class, random_state=random_state))
        .reset_index(level=0, drop=True)
    )
    remaining_data = df.drop(train_data.index)
    # print(train_data.index)

    return train_data, remaining_data


def sample_and_split(df_feat, df_label, ns=10, seed=100):
    # Check if n is greater than the minimum class size. If it is, raise an error.
    min_class_size = df_label.value_counts().min()
    if ns > min_class_size:
        raise ValueError(
            f"n is greater than your smallest class size of {min_class_size}."
        )

    # Sample n indices from each class
    sampled_indices = df_label.groupby(df_label).apply(
        lambda x: x.sample(ns, random_state=seed).index
    )

    # Concatenate indices from all classes into a single list
    sampled_indices = [idx for sublist in sampled_indices for idx in sublist]

    # Split df_feat into sampled indices and remaining
    sampled_df = df_feat.loc[sampled_indices]
    remaining_df = df_feat.drop(sampled_indices)

    # Split df_label into sampled indices and remaining
    sampled_labels = df_label.loc[sampled_indices]
    remaining_labels = df_label.drop(sampled_indices)

    return sampled_df, remaining_df, sampled_labels, remaining_labels


def evaluate_model(X_train, y_train, X_test, y_test, clf):

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # compute y_score
    try:
        y_score = clf.predict_proba(X_test)[:, 1]
    except:
        y_score = y_pred

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_score)
    except:
        auc = 0

    return acc, rec, prec, f1, auc, clf


def compute_synthcity_metrics(results, X_train_orig, y_train_orig, X_ref):

    from typing import Any, Tuple, Type

    # synthcity absolute
    from synthcity.metrics.eval_statistical import (
        AlphaPrecision,
        ChiSquaredTest,
        InverseKLDivergence,
        JensenShannonDistance,
        KolmogorovSmirnovTest,
        MaximumMeanDiscrepancy,
        PRDCScore,
        SurvivalKMDistance,
        WassersteinDistance,
    )

    from synthcity.metrics.eval_sanity import NearestSyntheticNeighborDistance

    from synthcity.plugins import Plugin, Plugins

    from synthcity.plugins.core.dataloader import (
        DataLoader,
        GenericDataLoader,
        create_from_info,
    )

    def _eval_plugin(
        evaluator_t: Type, X: DataLoader, X_syn: DataLoader, **kwargs: Any
    ) -> Tuple:
        evaluator = evaluator_t(**kwargs)

        syn_score = evaluator.evaluate(X, X_syn)

        return syn_score

    metrics = [
        AlphaPrecision,
        InverseKLDivergence,
        JensenShannonDistance,
        # KolmogorovSmirnovTest,
        MaximumMeanDiscrepancy,
        PRDCScore,
        WassersteinDistance,
        NearestSyntheticNeighborDistance,
    ]

    # easy_train, ambig_train, hard_train, Curator_llm  = data_centric(X_train_orig = X_train_orig,
    #             y_train_orig= y_train_orig,
    #             X_check = results['llm']['X'],
    #             y_check = results['llm']['y'])

    data_check_dict = {}

    for model in list(results.keys()):
        # tmp_dict = results[model]['X']
        # tmp_dict['y'] = results[model]['y']
        data_check_dict[f"{model}"] = results[model]["df"]

    # data_check_dict["X_train_llm_easy"] = results['llm']['X'].iloc[easy_train,:]
    # data_check_dict["X_train_llm_ambig"] = results['llm']['X'].iloc[ambig_train,:]
    # data_check_dict["X_train_llm_hard"] = results['llm']['X'].iloc[hard_train,:]

    statistical_metrics = {}

    for metric in metrics:
        # print(f"Metric: {metric}")
        tmp_dict = {}
        metric_name = metric.__name__
        for method in data_check_dict.keys():

            try:
                data_check = data_check_dict[method]

                if metric == AlphaPrecision:

                    if X_ref.shape[0] > data_check.shape[0]:
                        trial_results = _eval_plugin(
                            metric,
                            GenericDataLoader(
                                X_ref.astype(float).sample(data_check.shape[0])
                            ),
                            GenericDataLoader(data_check.astype(float)),
                        )
                    elif X_ref.shape[0] < data_check.shape[0]:
                        trial_results = _eval_plugin(
                            metric,
                            GenericDataLoader(X_ref.astype(float)),
                            GenericDataLoader(
                                data_check.astype(float).sample(X_ref.shape[0])
                            ),
                        )
                    else:
                        trial_results = _eval_plugin(
                            metric,
                            GenericDataLoader(X_ref.astype(float)),
                            GenericDataLoader(data_check.astype(float)),
                        )
                    # print(f"{method}: {trial_results}")
                else:

                    trial_results = _eval_plugin(
                        metric,
                        GenericDataLoader(X_ref.astype(float)),
                        GenericDataLoader(data_check.astype(float)),
                    )
                    # print(f"{method}: {trial_results}")

                if len(trial_results.keys()) == 1:
                    tmp_dict[method] = trial_results[list(trial_results.keys())[0]]
                else:
                    tmp_dict[method] = trial_results

            except Exception as e:
                import traceback

                print(traceback.format_exc())
                print(method, e)
                continue

        statistical_metrics[metric_name] = tmp_dict

    return statistical_metrics


def process_gpt(dataset, n_synthetic, temp, gpt_model, ns, seed):
    import pickle
    
    filename = f"../save_dfs/pipeline_llm_{dataset}_{n_synthetic}_{gpt_model}_{ns}_{seed}.pickle"
    if gpt_model == "gpt4_nocol":
        filename = f"../save_dfs/pipeline_llm_{dataset}_{n_synthetic}_gpt4_{ns}_{seed}_nocol.pickle"
    with open(filename, "rb") as f:
        loaded = pickle.load(f)

    if gpt_model == "gpt4_nocol":
        print(loaded["llm"].keys())
        df = loaded["llm"]["df"]

    else:
        df = loaded["llm"]["X"]
        df["target"] = loaded["llm"]["y"]

    df = df.dropna()
    df = df[
        ~df.apply(
            lambda row: any(
                [
                    isinstance(cell, str)
                    and cell
                    in [
                        "integer",
                        "float",
                        "numeric",
                        "categorical",
                        "number",
                        "No",
                        "Yes",
                        "continuous",
                        "age in years",
                        "string",
                    ]
                    for cell in row
                ]
            ),
            axis=1,
        )
    ]

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    if gpt_model == "gpt4_nocol":
        original_cols = example_df.columns
        example_df.columns = [
            "feat_" + str(i) for i in range(example_df.shape[1] - 1)
        ] + ["target"]

    try:
        df = df.astype(example_df.dtypes)
    except:
        # Assuming the dtypes from the example_df['Dtrain'].dataframe() is what you want
        target_dtypes = example_df.dtypes.to_dict()

        problematic_rows = set()

        for col, dtype in target_dtypes.items():
            for index, value in df[col].items():
                try:
                    _ = dtype.type(value)  # Try to convert the value
                except Exception:
                    problematic_rows.add(index)

        # Convert the problematic rows to a list and sort them
        problematic_rows = sorted(list(problematic_rows))

        # Drop the problematic rows
        df.drop(problematic_rows, inplace=True)

        # Identify rows where any cell is of type list
        rows_with_lists = df.applymap(lambda x: isinstance(x, list)).any(axis=1)

        # Drop those rows
        df = df[~rows_with_lists]

        df = df.astype(example_df.dtypes)

    if gpt_model == "gpt4_nocol":
        df.columns = original_cols
    return df


def compute_success(df):
    # Split the dataframe into two: easy variants and vanilla variants
    df_easy = df[df["Group"].str.contains("_easy")]
    df_vanilla = df[~df["Group"].str.contains("_easy")]

    # Initialize a counter
    counter = 0
    total = 0

    failures = []

    # Iterate over each row in the easy variant dataframe
    for _, row in df_easy.iterrows():
        # Extract the base name (i.e., without "_easy")
        base_name = row["Group"].replace("_easy", "")

        if base_name == "Oracle":
            continue

        # Extract the accuracy of the easy variant
        easy_accuracy = row["Accuracy"]

        # Extract the accuracy of the corresponding vanilla variant
        vanilla_accuracy = df_vanilla[df_vanilla["Group"] == base_name][
            "Accuracy"
        ].values[0]

        # Check if the easy variant has better performance
        if easy_accuracy > vanilla_accuracy:
            counter += 1
        else:
            failures.append(base_name)

        total += 1

    return counter, total, failures


def process_llama(
    dataset, n_synthetic, temp, llama_model, ns, seed, path="./llama-gen/llama-data"
):
    import pickle

    filename = f"{path}/{llama_model}_{dataset}_{n_synthetic}_{ns}_{seed}.pickle"

    with open(filename, "rb") as f:
        loaded = pickle.load(f)

    df = loaded["llm"]["X"]
    df["target"] = loaded["llm"]["y"]

    # df = df.dropna()
    df = df[
        ~df.apply(
            lambda row: any(
                [
                    isinstance(cell, str)
                    and cell
                    in [
                        "integer",
                        "float",
                        "numeric",
                        "categorical",
                        "number",
                        "No",
                        "Yes",
                        "continuous",
                        "age in years",
                        "string",
                    ]
                    for cell in row
                ]
            ),
            axis=1,
        )
    ]

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    if dataset == "compas" and llama_model == "llama13b":
        # Convert lists in the 'Sex' column to single values (strings)
        df = df[df["sex"].apply(lambda x: not isinstance(x, list))]
        # Define the mapping dictionary
        sex_mapping = {"Male": 1, "Female": 0}
        # Apply the mapping to the 'Sex' column
        df["sex"] = df["sex"].map(sex_mapping)

    if dataset == "adult" and llama_model == "llama13b":
        df = df[df["age"].apply(lambda x: not isinstance(x, list))]

        # Define a custom function to set the values based on conditions
        def set_target_value(target_value):
            try:
                target_value = float(target_value)
                if target_value > 1 and target_value < 50000:
                    return 0
                elif target_value >= 50000:
                    return 1
                else:
                    return target_value  # Keep the original value if it doesn't meet the conditions
            except (ValueError, TypeError):
                return None  # Return None for rows where the conversion to float fails

        # Apply the custom function to update the 'Target' column
        df["target"] = df["target"].apply(set_target_value)

        # Drop rows where 'Target' is None
        df = df.dropna(subset=["target"])

    try:
        df = df.astype(example_df.dtypes)
    except:
        # Assuming the dtypes from the example_df['Dtrain'].dataframe() is what you want
        target_dtypes = example_df.dtypes.to_dict()

        problematic_rows = set()

        for col, dtype in target_dtypes.items():
            for index, value in df[col].items():
                try:
                    _ = dtype.type(value)  # Try to convert the value
                except Exception:
                    problematic_rows.add(index)

        # Convert the problematic rows to a list and sort them
        problematic_rows = sorted(list(problematic_rows))

        # Drop the problematic rows
        df.drop(problematic_rows, inplace=True)

        # Identify rows where any cell is of type list
        rows_with_lists = df.applymap(lambda x: isinstance(x, list)).any(axis=1)

        # Drop those rows
        df = df[~rows_with_lists]

        df = df.astype(example_df.dtypes)

    return df


def process_together(dataset, n_synthetic, temp, gpt_model, ns, seed):
    import pickle

    filename = f"./together_dfs/pipeline_llm_{dataset}_{n_synthetic}_{gpt_model}_{ns}_{seed}.pickle"

    with open(filename, "rb") as f:
        loaded = pickle.load(f)

    # df = loaded['llm']['df']

    df = loaded["llm"]["X"]
    y_tmp = loaded["llm"]["y"]

    df["y"] = y_tmp

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    df.loc[df["target"].isna(), "target"] = df.loc[df["target"].isna(), "y"]
    df = df[example_df.columns]

    # df = df.dropna()
    df = df[
        ~df.apply(
            lambda row: any(
                [
                    isinstance(cell, str)
                    and cell
                    in [
                        "integer",
                        "float",
                        "numeric",
                        "categorical",
                        "number",
                        "No",
                        "Yes",
                        "continuous",
                        "age in years",
                        "string",
                    ]
                    for cell in row
                ]
            ),
            axis=1,
        )
    ]

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    if gpt_model == "gpt4_nocol":
        original_cols = example_df.columns
        print(example_df.shape)
        example_df.columns = [
            "feat_" + str(i) for i in range(example_df.shape[1] - 1)
        ] + ["target"]

    try:
        df = df.astype(example_df.dtypes)
    except:
        # Assuming the dtypes from the example_df['Dtrain'].dataframe() is what you want
        target_dtypes = example_df.dtypes.to_dict()

        problematic_rows = set()

        for col, dtype in target_dtypes.items():
            for index, value in df[col].items():
                try:
                    _ = dtype.type(value)  # Try to convert the value
                except Exception:
                    problematic_rows.add(index)

        # Convert the problematic rows to a list and sort them
        problematic_rows = sorted(list(problematic_rows))

        # Drop the problematic rows
        df.drop(problematic_rows, inplace=True)

        # Identify rows where any cell is of type list
        rows_with_lists = df.applymap(lambda x: isinstance(x, list)).any(axis=1)

        # Drop those rows
        df = df[~rows_with_lists]

        df = df.astype(example_df.dtypes)

    return df


def process_swahili(dataset, n_synthetic, temp, gpt_model, ns, seed):
    import pickle

    filename = f"./swahili_dfs/pipeline_llm_{dataset}_{n_synthetic}_{gpt_model}_{ns}_{seed}.pickle"

    with open(filename, "rb") as f:
        loaded = pickle.load(f)

    if gpt_model == "gpt4_nocol":
        print(loaded["llm"].keys())
        df = loaded["llm"]["df"]

    else:
        df = loaded["llm"]["X"]
        df["target"] = loaded["llm"]["y"]

    df = df.dropna()
    df = df[
        ~df.apply(
            lambda row: any(
                [
                    isinstance(cell, str)
                    and cell
                    in [
                        "integer",
                        "float",
                        "numeric",
                        "categorical",
                        "number",
                        "No",
                        "Yes",
                        "continuous",
                        "age in years",
                        "string",
                    ]
                    for cell in row
                ]
            ),
            axis=1,
        )
    ]

    example_df = loaded["Original"]["X"]
    example_df["target"] = loaded["Original"]["y"]

    df = df[example_df.columns]

    if gpt_model == "gpt4_nocol":
        original_cols = example_df.columns
        print(example_df.shape)
        example_df.columns = [
            "feat_" + str(i) for i in range(example_df.shape[1] - 1)
        ] + ["target"]

    try:
        df = df.astype(example_df.dtypes)
    except:
        # Assuming the dtypes from the example_df['Dtrain'].dataframe() is what you want
        target_dtypes = example_df.dtypes.to_dict()

        problematic_rows = set()

        for col, dtype in target_dtypes.items():
            for index, value in df[col].items():
                try:
                    _ = dtype.type(value)  # Try to convert the value
                except Exception:
                    problematic_rows.add(index)

        # Convert the problematic rows to a list and sort them
        problematic_rows = sorted(list(problematic_rows))

        # Drop the problematic rows
        df.drop(problematic_rows, inplace=True)

        # Identify rows where any cell is of type list
        rows_with_lists = df.applymap(lambda x: isinstance(x, list)).any(axis=1)

        # Drop those rows
        df = df[~rows_with_lists]

        df = df.astype(example_df.dtypes)

    return df
