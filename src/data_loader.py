# Imports
from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost as xgb
from torch.utils.data import DataLoader, Dataset

import openml
import pandas as pd
from openml.datasets import edit_dataset, fork_dataset, get_dataset


def str2num(s, encoder):
    return encoder[s]


def load_seer_cutract_dataset(name, seed, prop=1.0):
    """
    It loads the SEER/CUTRACT dataset, and returns the features, labels, and the entire dataset

    Args:
      name: the name of the dataset to load.
      seed: the random seed used to generate the data

    Returns:
      The features, labels, and the entire dataset.
    """

    def aggregate_grade(row):
        if row["grade_1.0"] == 1:
            return 1
        if row["grade_2.0"] == 1:
            return 2
        if row["grade_3.0"] == 1:
            return 3
        if row["grade_4.0"] == 1:
            return 4
        if row["grade_5.0"] == 1:
            return 5

    def aggregate_stage(row):
        if row["stage_1"] == 1:
            return 1
        if row["stage_2"] == 1:
            return 2
        if row["stage_3"] == 1:
            return 3
        if row["stage_4"] == 1:
            return 4
        if row["stage_5"] == 1:
            return 5

    random_seed = seed

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade",
    ]

    features = [
        "age",
        "psa",
        "comorbidities",
        "treatment_CM",
        "treatment_Primary hormone therapy",
        "treatment_Radical Therapy-RDx",
        "treatment_Radical therapy-Sx",
        "grade",
        "stage_1",
        "stage_2",
        "stage_3",
        "stage_4",
    ]

    label = "mortCancer"
    df = pd.read_csv(f"./data/{name}.csv")

    df["grade"] = df.apply(aggregate_grade, axis=1)
    df["stage"] = df.apply(aggregate_stage, axis=1)
    df["mortCancer"] = df["mortCancer"].astype(int)
    df["mort"] = df["mort"].astype(int)

    mask = df[label] == True
    df_dead = df[mask]
    df_survive = df[~mask]

    if name == "seer":
        total_balanced = 10000
        n_samples_per_group = int(prop * total_balanced)

        n_samples = n_samples_per_group  # 1000 #1000-
        ns = n_samples_per_group  # 1000 #10000
    else:
        total_balanced = 1000
        n_samples_per_group = int(prop * total_balanced)

        n_samples = n_samples_per_group
        ns = n_samples_per_group
    df = pd.concat(
        [
            df_dead.sample(ns, random_state=random_seed),
            df_survive.sample(n_samples, random_state=random_seed),
        ]
    )
    df = sklearn.utils.shuffle(df, random_state=random_seed)
    df = df.reset_index(drop=True)
    return df[features], df[label], df


def load_adult_dataset(prop=1, seed=42):
    """
    > This function loads the adult dataset, removes all the rows with missing values, and then splits the data into
    a training and test set
    Args:
      split_size: The proportion of the dataset to include in the test split.
    Returns:
      X_train, X_test, y_train, y_test, X, y
    """

    def process_dataset(df):
        """
        > This function takes a dataframe, maps the categorical variables to numerical values, and returns a
        dataframe with the numerical values
        Args:
          df: The dataframe to be processed
        Returns:
          a dataframe after the mapping
        """

        data = [df]

        salary_map = {" <=50K": 1, " >50K": 0}
        df["salary"] = df["salary"].map(salary_map).astype(int)

        df["sex"] = df["sex"].map({" Male": 1, " Female": 0}).astype(int)

        df["country"] = df["country"].replace(" ?", np.nan)
        df["workclass"] = df["workclass"].replace(" ?", np.nan)
        df["occupation"] = df["occupation"].replace(" ?", np.nan)

        df.dropna(how="any", inplace=True)

        for dataset in data:
            dataset.loc[
                dataset["country"] != " United-States",
                "country",
            ] = "Non-US"
            dataset.loc[
                dataset["country"] == " United-States",
                "country",
            ] = "US"

        df["country"] = df["country"].map({"US": 1, "Non-US": 0}).astype(int)

        df["marital-status"] = df["marital-status"].replace(
            [
                " Divorced",
                " Married-spouse-absent",
                " Never-married",
                " Separated",
                " Widowed",
            ],
            "Single",
        )
        df["marital-status"] = df["marital-status"].replace(
            [" Married-AF-spouse", " Married-civ-spouse"],
            "Couple",
        )

        df["marital-status"] = df["marital-status"].map(
            {"Couple": 0, "Single": 1},
        )

        rel_map = {
            " Unmarried": 0,
            " Wife": 1,
            " Husband": 2,
            " Not-in-family": 3,
            " Own-child": 4,
            " Other-relative": 5,
        }

        df["relationship"] = df["relationship"].map(rel_map)

        race_map = {
            " White": 0,
            " Amer-Indian-Eskimo": 1,
            " Asian-Pac-Islander": 2,
            " Black": 3,
            " Other": 4,
        }

        df["race"] = df["race"].map(race_map)

        def f(x):
            if (
                x["workclass"] == " Federal-gov"
                or x["workclass"] == " Local-gov"
                or x["workclass"] == " State-gov"
            ):
                return "govt"
            elif x["workclass"] == " Private":
                return "private"
            elif (
                x["workclass"] == " Self-emp-inc"
                or x["workclass"] == " Self-emp-not-inc"
            ):
                return "self_employed"
            else:
                return "without_pay"

        df["employment_type"] = df.apply(f, axis=1)

        employment_map = {
            "govt": 0,
            "private": 1,
            "self_employed": 2,
            "without_pay": 3,
        }

        df["employment_type"] = df["employment_type"].map(employment_map)
        df.drop(
            labels=[
                "workclass",
                "education",
                "occupation",
            ],
            axis=1,
            inplace=True,
        )
        df.loc[(df["capital-gain"] > 0), "capital-gain"] = 1
        df.loc[(df["capital-gain"] == 0, "capital-gain")] = 0

        df.loc[(df["capital-loss"] > 0), "capital-loss"] = 1
        df.loc[(df["capital-loss"] == 0, "capital-loss")] = 0

        return df

    try:
        df = pd.read_csv("data/adult.csv", delimiter=",")
    except BaseException:
        df = pd.read_csv("../data/adult.csv", delimiter=",")

    df = process_dataset(df)
    df = df.sample(frac=prop, random_state=seed)

    return (df.drop(["salary"], axis=1), df["salary"], df)


def load_support_dataset(prop=1, seed=42):
    """Load the Support dataset"""
    df = pd.read_csv("./data/support_data.csv")
    df = df.drop(columns=["Unnamed: 0", "d.time"])
    df = df.sample(frac=prop, random_state=seed)
    return (df.drop(columns=["death"]), df["death"], df)

def load_maggic_dataset(prop=1, seed=42):
    df = pd.read_csv('data/Maggic.csv')
    df = df.drop(columns=['days_to_fu'])

    df = df.sample(frac=prop, random_state=seed)
    return (df.drop(columns=["death_all"]), df["death_all"], df)


def load_covid_dataset(prop=1, seed=42):
    """Load the Covid dataset"""
    df = pd.read_csv("./data/covid.csv")
    x_ids = [
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
    ]

    df = df.sample(frac=prop, random_state=seed)

    return (df.iloc[:, x_ids].drop(columns=["Race", "SG_UF_NOT"]), df.iloc[:, 0], df)


def load_metabric(prop=1, seed=42):

    def metabric_preprocess(positive_string, negative_string):
        def preprocess(x):
            x = str(x)
            if x == negative_string:
                return 0
            elif x == positive_string:
                return 1
            else:
                return -1

        return preprocess


    metabric_info = {
        1: {"name": "pr_status", "preprocess": metabric_preprocess("Positive", "Negative")},
        2: {"name": "er_status", "preprocess": metabric_preprocess("Positive", "Negative")},
        3: {
            "name": "er_status_measured_by_ihc",
            "preprocess": metabric_preprocess("Positve", "Negative"),
        },
        4: {
            "name": "her2_status",
            "preprocess": metabric_preprocess("Positive", "Negative"),
        },
        5: {
            "name": "her2_status_measured_by_snp6",
            "preprocess": metabric_preprocess("GAIN", "NEUTRAL"),
        },
    }


    raw_data = pd.read_csv(
                    "data/METABRIC_RNA_Mutation.csv",
                    low_memory=False,
                )
            
    rule = 1
    Data = pd.DataFrame(raw_data.iloc[:, 31:520])
    y_data = pd.DataFrame(raw_data[metabric_info[rule]["name"]])
    y_data = np.array(
        y_data[metabric_info[rule]["name"]].apply(
            metabric_info[rule]["preprocess"],
        ),
    )

    Data['cohort'] = raw_data.cohort
    Data['y'] = y_data

    Data = Data.sample(frac=prop, random_state=seed)

    return Data.drop('y', axis=1), Data['y'], Data

def load_cover(seed=42,prop=1):
    df = pd.read_csv("data/covtype.csv")

    df.rename(columns={'Cover_Type': 'y'}, inplace=True)

    df = df.sample(frac=prop, random_state=seed)

    return df.drop('y', axis=1), df['y'], df

def load_contraceptive(seed=42,prop=1):
    df = pd.read_csv("data/cmc.data", delimiter=',')
    df.rename(columns={'1.2': 'y'}, inplace=True)
    df = df.sample(frac=prop, random_state=seed)

    df['y'] = df['y'].replace({1: 0, 2: 1, 3:2})

    return df.drop('y', axis=1), df['y'], df

def load_credit(seed=42,prop=1):
    df = pd.read_excel('data/credit_default.xls', header=1)

    df.drop(['ID'], axis=1, inplace=True)

    # rename columns default payment next month to y
    df.rename(columns={'default payment next month': 'y'}, inplace=True)

    return df.drop('y', axis=1), df['y'], df


def load_telescope(seed=42, prop=1):
    id = 1120 # 20k
    dataset = openml.datasets.get_dataset(id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    df["y"] = y


    df = df.sample(frac=prop, random_state=seed)

    return df.drop('y', axis=1), df['y'], df

def load_marketing(seed=42, prop=1):
    id = 1461 # 45k
    dataset = openml.datasets.get_dataset(id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    df["y"] = y


    df = df.sample(frac=prop, random_state=seed)

    return df.drop('y', axis=1), df['y'], df

def load_compas(seed=42, prop=1):
    id = 42192 # 5k
    dataset = openml.datasets.get_dataset(id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    df["y"] = y


    df = df.sample(frac=prop, random_state=seed)

    return df.drop('y', axis=1), df['y'], df

def load_eye(seed=42, prop=1):
    id = 1044 # 11k
    dataset = openml.datasets.get_dataset(id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    df["y"] = y


    df = df.sample(frac=prop, random_state=seed)

    return df.drop('y', axis=1), df['y'], df

def load_bio(seed=42, prop=1):
    id = 4134 # 3k
    dataset = openml.datasets.get_dataset(id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    df["y"] = y


    df = df.sample(frac=prop, random_state=seed)

    return df.drop('y', axis=1), df['y'], df




# def load_credit(seed=42,prop=1):
#     #import sklear std scaler and labelencoder
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.preprocessing import LabelEncoder
#     file = './data/german.data'
#     url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

#     names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 
#             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 
#             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 
#             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

#     data = pd.read_csv(file,names = names, delimiter=' ')
#     data.classification.replace([1,2], [1,0], inplace=True)

#     #numerical variables labels
#     numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 
#             'existingcredits', 'peopleliable', 'classification']

#     # Standardization
#     numdata_std = pd.DataFrame(StandardScaler().fit_transform(data[numvars].drop(['classification'], axis=1)))

#     from collections import defaultdict

#     #categorical variables labels
#     catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
#             'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 
#             'telephone', 'foreignworker']

#     d = defaultdict(LabelEncoder)

#     # Encoding the variable
#     lecatdata = data[catvars].apply(lambda x: d[x.name].fit_transform(x))



#     #One hot encoding, create dummy variables for every category of every categorical variable
#     dummyvars = pd.get_dummies(data[catvars])

#     df = pd.concat([data[numvars], dummyvars], axis = 1)

#     # replace classification with y
#     df.rename(columns={'classification': 'y'}, inplace=True)
#     return df.drop('y', axis=1), df['y'], df
    

def load_higgs(seed=42, prop=1):
    df = pd.read_csv("data/training.csv")

    df.drop(columns=["EventId"], inplace=True)

    drop_columns=['DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet','DER_lep_eta_centrality','PRI_jet_subleading_pt','PRI_jet_subleading_eta','PRI_jet_subleading_phi', "Weight"]

    df.drop(columns=drop_columns, inplace=True)

    df['Label'] = df['Label'].replace({'s': 0, 'b': 1})

    df.rename(columns={'Label': 'y'}, inplace=True)

    df = df.sample(frac=prop, random_state=seed)

    return df.drop('y', axis=1), df['y'], df


def load_blog_data(seed=42, prop=1):
    df = pd.read_csv("data/blogData_train.csv", header=None)

    df.rename(columns={280: 'y'}, inplace=True)

    df['y'] = df['y']>0
    df['y'] = df['y'].astype(int)

    df = df.sample(frac=prop, random_state=seed)

    return df.drop('y', axis=1), df['y'], df


def load_bank(seed=42, prop=1):
    from sklearn.utils import shuffle
    df = pd.read_csv('data/VariantI.csv')
    for col in ["payment_type", "employment_status", "housing_status", "source", "device_os"]:
        df[col] = df[col].astype("category").cat.codes

    df.rename(columns={'fraud_bool': 'y'}, inplace=True)

    mask = df['y'] == True
    df_fraud = df[mask]
    df_no = df[~mask]

    n_samples = 10000
    df = pd.concat(
        [
            df_fraud.sample(n_samples, random_state=seed),
            df_no.sample(n_samples, random_state=seed),
        ]
    )

    df = shuffle(df, random_state=seed)
    df = df.sample(frac=prop, random_state=seed)

    return df.drop('y', axis=1), df['y'], df


def load_drug_dataset(prop=1, seed=42):
    from sklearn.impute import SimpleImputer
    data = pd.read_csv('data/Drug_Consumption.csv')

    #Drop overclaimers, Semer, and other nondrug columns
    data = data.drop(data[data['Semer'] != 'CL0'].index)
    data = data.drop(['Semer', 'Caff', 'Choc'], axis=1)
    data.reset_index()
    print(f'In the new dataframe there are {data.shape[0]} rows and {data.shape[1]} columns')

    # Binary encode gender
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)

    # Encode ordinal features
    ordinal_features = ['Age', 
                        'Education',
                        'Alcohol',
                        'Amyl',
                        'Amphet',
                        'Benzos',
                        'Cannabis',
                        'Coke',
                        'Crack',
                        'Ecstasy',
                        'Heroin',
                        'Ketamine',
                        'Legalh',
                        'LSD',
                        'Meth',
                        'Mushrooms',
                        'Nicotine',
                        'VSA'    ]

    # Define ordinal orderings
    ordinal_orderings = [
        ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        ['Left school before 16 years', 
        'Left school at 16 years', 
        'Left school at 17 years', 
        'Left school at 18 years',
        'Some college or university, no certificate or degree',
        'Professional certificate/ diploma',
        'University degree',
        'Masters degree',
        'Doctorate degree'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6'],
        ['CL0','CL1','CL2','CL3','CL4','CL5','CL6']
    ]

    # Nominal features
    nominal_features = ['Country',
                        'Ethnicity']

    #Create function for ordinal encoding
    def ordinal_encoder(df, columns, ordering):
        df = df.copy()
        for column, ordering in zip(ordinal_features, ordinal_orderings):
            df[column] = df[column].apply(lambda x: ordering.index(x))
        return df

    def cat_converter(df, columns):
        df = df.copy()
        for column in columns:
            df[column] = df[column].astype('category').cat.codes
        return df

    data = ordinal_encoder(data, ordinal_features, ordinal_orderings)
    data = cat_converter(data, nominal_features)


    nic_df = data.copy()
    nic_df['y'] = nic_df['Nicotine'].apply(lambda x: 1 if x not in [0,1] else 0)
    nic_df = nic_df.drop(['ID','Nicotine'], axis=1)

    return nic_df.drop('y', axis=1), nic_df['y'], nic_df

def load_fraud_dataset(prop=1, seed=42):
    df = pd.read_csv('data/creditcard.csv')
    df = df.sample(frac=prop, random_state=seed)
    return (df.drop(columns=["Class"]), df["Class"], df)


def get_data(dataset, seed=42, prop=1.0):
    """
    It takes in the training data and labels, and the number of estimators, and returns the indices of
    the easy, inconsistent, and hard training examples.

    Args:
      dataset: the name of the dataset you want to use.
      seed: the random seed used to split the data into train and test sets. Defaults to 42
    """
    from sklearn.model_selection import train_test_split

    assert dataset in [
        "seer",
        "cutract",
        "covid",
        "support",
        "adult",
        "bank",
        "metabric",
        "drug",
        "maggic",
        "fraud",
        "higgs",
        "contraceptive",
        "blog",
        "cover",
        "credit",
        "telescope",
        "bio",
        "eye",
        "compas",
        "marketing",
    ], f"The dataset {dataset} not supported yet..."

    if dataset in ["seer", "cutract"]:
        df_feat, df_label, df = load_seer_cutract_dataset(dataset, seed=seed, prop=prop)

    if dataset == "covid":
        df_feat, df_label, df = load_covid_dataset(prop=prop, seed=seed)

    if dataset == "support":
        df_feat, df_label, df = load_support_dataset(prop=prop, seed=seed)

    if dataset == "adult":
        df_feat, df_label, df = load_adult_dataset(prop=prop, seed=seed)

    if dataset == "metabric":
        df_feat, df_label, df = load_metabric(prop=prop, seed=seed)

    if dataset == "bank":
        df_feat, df_label, df = load_bank(prop=prop, seed=seed)

    if dataset == "drug":
        df_feat, df_label, df = load_drug_dataset(prop=prop, seed=seed)

    if dataset == "maggic":
        df_feat, df_label, df = load_maggic_dataset(prop=prop, seed=seed)

    if dataset == "fraud":
        df_feat, df_label, df = load_fraud_dataset(prop=prop, seed=seed)

    if dataset == "higgs":
        df_feat, df_label, df = load_higgs(prop=prop, seed=seed)
    
    if dataset == "contraceptive":
        df_feat, df_label, df = load_contraceptive(prop=prop, seed=seed)

    if dataset == "blog":
        df_feat, df_label, df = load_blog_data(prop=prop, seed=seed)

    if dataset == "cover":
        df_feat, df_label, df = load_cover(prop=prop, seed=seed)

    if dataset == "credit":
        df_feat, df_label, df = load_credit(prop=prop, seed=seed)

    if dataset == "telescope":
        df_feat, df_label, df = load_telescope(prop=prop, seed=seed)

    if dataset == "bio":
        df_feat, df_label, df = load_bio(prop=prop, seed=seed)

    if dataset == "eye":
        df_feat, df_label, df = load_eye(prop=prop, seed=seed)

    if dataset == "compas":
        df_feat, df_label, df = load_compas(prop=prop, seed=seed)

    if dataset == "marketing":
        df_feat, df_label, df = load_marketing(prop=prop, seed=seed)

    return df_feat, df_label, df

