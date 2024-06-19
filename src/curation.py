import math
import tempfile
from contextlib import suppress
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch


class Curator:
    def __init__(self, X, y, sparse_labels: bool = False, catboost: bool = False):
        """
        The function takes in the training data and the labels, and stores them in the class variables X
        and y. It also stores the boolean value of sparse_labels in the class variable _sparse_labels
        Args:
          X: the input data
          y: the true labels
          sparse_labels (bool): boolean to identify if labels are one-hot encoded or not. If not=True.
        Defaults to False
        """
        self.X = X
        self.y = np.asarray(y).tolist()
        self._sparse_labels = sparse_labels

        # placeholder
        self._gold_labels_probabilities = None
        self._true_probabilities = None
        self.catboost = catboost

    def on_epoch_end(self, clf, device="cpu", iteration=1, **kwargs):
        """
        The function computes the gold label and true label probabilities over all samples in the
        dataset
        We iterate through the dataset, and for each sample, we compute the gold label probability (i.e.
        the actual ground truth label) and the true label probability (i.e. the predicted label).
        We then append these probabilities to the `_gold_labels_probabilities` and `_true_probabilities`
        lists.
        We do this for every sample in the dataset
        Args:
          clf: the classifier object
          device: The device to use for the computation. Defaults to cpu
          iteration: The current iteration of the training loop. Defaults to 1
        """
        # Compute both the gold label and true label probabilities over all samples in the dataset
        gold_label_probabilities = (
            list()
        )  # gold label probabilities, i.e. actual ground truth label
        true_probabilities = list()  # true label probabilities, i.e. predicted label

        x = self.X
        y = torch.tensor(self.y, device=device)

        if len(y.shape) == 2:
            y = y.squeeze()

        if not self.catboost:
            probabilities = torch.tensor(
                clf.predict_proba(x, iteration_range=(0, iteration)),
                device=device,
            )

        else:
            probabilities = torch.tensor(
                clf.predict_proba(x, ntree_start=0, ntree_end=iteration),
                device=device,
            )

        # one hot encode the labels
        y = torch.nn.functional.one_hot(
            y.to(torch.int64),
            num_classes=probabilities.shape[-1],
        )

        # Now we extract the gold label and predicted true label probas

        # If the labels are binary [0,1]
        if len(torch.squeeze(y)) == 1:
            # get true labels
            true_probabilities = torch.tensor(probabilities)

            # get gold labels
            probabilities, y = torch.squeeze(
                torch.tensor(probabilities),
            ), torch.squeeze(y)

            batch_gold_label_probabilities = torch.where(
                y == 0,
                1 - probabilities,
                probabilities,
            )

        # if labels are one hot encoded, e.g. [[1,0,0], [0,1,0]]
        elif len(torch.squeeze(y)) == 2:
            # get true labels
            batch_true_probabilities = torch.max(probabilities)

            # get gold labels
            batch_gold_label_probabilities = torch.masked_select(
                probabilities,
                y.bool(),
            )
        else:

            # get true labels
            batch_true_probabilities = torch.max(probabilities)

            # get gold labels
            batch_gold_label_probabilities = torch.masked_select(
                probabilities,
                y.bool(),
            )

        # move torch tensors to cpu as np.arrays()
        batch_gold_label_probabilities = batch_gold_label_probabilities.cpu().numpy()
        batch_true_probabilities = batch_true_probabilities.cpu().numpy()

        # Append the new probabilities for the new batch
        gold_label_probabilities = np.append(
            gold_label_probabilities,
            [batch_gold_label_probabilities],
        )
        true_probabilities = np.append(true_probabilities, [batch_true_probabilities])

        # Append the new gold label probabilities
        if self._gold_labels_probabilities is None:  # On first epoch of training
            self._gold_labels_probabilities = np.expand_dims(
                gold_label_probabilities,
                axis=-1,
            )
        else:
            stack = [
                self._gold_labels_probabilities,
                np.expand_dims(gold_label_probabilities, axis=-1),
            ]
            self._gold_labels_probabilities = np.hstack(stack)

        # Append the new true label probabilities
        if self._true_probabilities is None:  # On first epoch of training
            self._true_probabilities = np.expand_dims(true_probabilities, axis=-1)
        else:
            stack = [
                self._true_probabilities,
                np.expand_dims(true_probabilities, axis=-1),
            ]
            self._true_probabilities = np.hstack(stack)

    @property
    def gold_labels_probabilities(self) -> np.ndarray:
        """
        Returns:
            Gold label predicted probabilities of the "gold" label: np.array(n_samples, n_epochs)
        """
        return self._gold_labels_probabilities

    @property
    def true_probabilities(self) -> np.ndarray:
        """
        Returns:
            Actual predicted probabilities of the predicted label: np.array(n_samples, n_epochs)
        """
        return self._true_probabilities

    @property
    def confidence(self) -> np.ndarray:
        """
        Returns:
            Average predictive confidence across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities, axis=-1)

    @property
    def aleatoric(self):
        """
        Returns:
            Aleatric uncertainty of true label probability across epochs: np.array(n_samples): np.array(n_samples)
        """
        preds = self._gold_labels_probabilities
        return np.mean(preds * (1 - preds), axis=-1)

    @property
    def variability(self) -> np.ndarray:
        """
        Returns:
            Epistemic variability of true label probability across epochs: np.array(n_samples)
        """
        return np.std(self._gold_labels_probabilities, axis=-1)

    @property
    def correctness(self) -> np.ndarray:
        """
        Returns:
            Proportion of times a sample is predicted correctly across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities > 0.5, axis=-1)

    @property
    def entropy(self):
        """
        Returns:
            Predictive entropy of true label probability across epochs: np.array(n_samples)
        """
        X = self._gold_labels_probabilities
        return -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

    @property
    def mi(self):
        """
        Returns:
            Mutual information of true label probability across epochs: np.array(n_samples)
        """
        X = self._gold_labels_probabilities
        entropy = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

        X = np.mean(self._gold_labels_probabilities, axis=1)
        entropy_exp = -1 * np.sum(X * np.log(X + 1e-12), axis=-1)
        return entropy - entropy_exp






def get_groups(confidence, aleatoric_uncertainty, curation_xthresh, curation_ythresh):
    import numpy as np
    percentile_thresh = 50
    thresh = curation_ythresh
    conf_thresh_low = curation_ythresh
    conf_thresh_high = 1 - curation_ythresh
    conf_thresh = 0.5

    #x_thresh = np.percentile(aleatoric_uncertainty, curation_xthresh) #curation_xthresh
    x_thresh = curation_xthresh
    
    hard_train = np.where(
        (confidence <= conf_thresh_low) & (aleatoric_uncertainty <= x_thresh)
    )[0]
   
    easy_train = np.where(
        (confidence >= conf_thresh_high) & (aleatoric_uncertainty <= x_thresh)
    )[0]

    hard_easy = np.concatenate((hard_train, easy_train))
    ambig_train = []
    for id in range(len(confidence)):
        if id not in hard_easy:
            ambig_train.append(id)

    ambig_train = np.array(ambig_train)
    
    return easy_train, ambig_train, hard_train



def data_centric_curation(X_train_orig, y_train_orig, X_check, y_check, 
                 curation_metric="aleatoric", retrain=False, nest = 100, 
                 curation_ythresh=0.2, curation_xthresh=0.2):
    from xgboost import XGBClassifier
    import numpy as np

    # train xgboost on X_train_orig, y_train_orig 
    xgb = XGBClassifier(n_estimators=nest)
    xgb.fit(X_train_orig, y_train_orig)
    if retrain:
        xgb = XGBClassifier(n_estimators=nest)
        xgb.fit(X_check, y_check)


    Curator_xgb = Curator(X=X_check, y=y_check)

    for i in range(1, nest):
        # *** Characterize with Curator [LINE 2] ***
        Curator_xgb.on_epoch_end(clf=xgb, iteration=i)

    if curation_metric == "aleatoric":
        curation_xmetric = Curator_xgb.aleatoric
    elif curation_metric == "epistemic":
        curation_xmetric = Curator_xgb.variability
    elif curation_metric == "entropy":
        curation_xmetric = Curator_xgb.entropy
    elif curation_metric == "mi":
        curation_xmetric = Curator_xgb.mi

    confidence = Curator_xgb.confidence
    # confidence is an array of size [N,1] where N is the number of training data points
    
    if curation_xthresh==0:
        print('Using adaptive threshold')
        curation_xthresh = 0.75 * (np.max(curation_xmetric) - np.min(curation_xmetric))
    curation_ythresh = curation_ythresh

    curated_train, ambig_train, unlearnable_train = get_groups(
        confidence=confidence,
        aleatoric_uncertainty=curation_xmetric,
        curation_xthresh=curation_xthresh,
        curation_ythresh=curation_ythresh,
    )

    return curated_train, ambig_train, unlearnable_train, Curator_xgb

    