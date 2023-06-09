import pandas as pd
import numpy as np
import re


def load_all():
    train_df = pd.read_csv('recsys_data/train.csv')
    articles_df = pd.read_csv('recsys_data/articles.csv')
    users_df = pd.read_csv('recsys_data/users.csv')
    test_df = pd.read_csv('recsys_data/test.csv')
    results_df = pd.read_csv('recsys_data/results.csv')
    return train_df, articles_df, users_df, test_df, results_df


def get_embeddings(model,text):
    """
    Split texts into sentences and get embeddings for each sentence.
    The final embeddings is the mean of all sentence embeddings.
    :param text: str. Input text.
    :return: np.array. Embeddings.
    """
    return np.mean(
        model.encode(
            list(set(re.findall('[^!?。.？！]+[!?。.？！]?', text)))
        ), axis=0)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])