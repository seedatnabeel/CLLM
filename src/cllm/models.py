from tqdm import tqdm
from synthcity.plugins import Plugins
from imblearn.over_sampling import SMOTE


def synthcity_generate(name, X_train, y_train, n_synthetic=1000):

    # apply synthetic data to training data
    syn_model = Plugins().get(name)
    X_train["target"] = y_train
    syn_model.fit(X_train)
    X_train_syn = syn_model.generate(count=n_synthetic).dataframe()

    return X_train_syn.drop(columns=["target"]), X_train_syn["target"]


def smote(X_train, y_train, n_synthetic=1000):

    sampling_strategy = {0: n_synthetic, 1: n_synthetic}

    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=4)
    X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

    return X_train_smote, y_train_smote


def great(X_train, y_train, n_synthetic=1000, epochs=1000, max_length=4000):
    from be_great import GReaT

    model = GReaT(llm="distilgpt2", batch_size=64, epochs=epochs, save_steps=400000)

    X_train["target"] = y_train

    model.fit(X_train.astype("float16"))
    # model.fit(X_train)

    X_train_syn = model.sample(n_samples=n_synthetic, max_length=max_length)

    # get target column as y_train
    y_train_great = X_train_syn["target"]

    # drop target column from X_train_syn
    X_train_great = X_train_syn.drop(columns=["target"])

    return X_train_great, y_train_great
