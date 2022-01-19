import logging
import os
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def create_vocabulary(series: pd.Series,
                      *args: str,
                      **kwargs: int) -> Tuple[List, Dict]:
    """
    Creates a vocabulary from the words contained in the values of a pandas series.

    :param series: A pandas series containing text data.
    :param args: Non-keyword varargs.
    :param kwargs: Keyword varargs.

    :return: A tuple containing a list (vocabulary) and a dict (mapping from word to index in vocab).
    """
    counts = Counter()
    for _, value in series.iteritems():
        content_words = [token for token in value.split()]
        counts.update(content_words)

    ignore_rare_words = kwargs.get("ignore_rare_words", False)
    if ignore_rare_words:
        logger.info("Removing rare words...")
        rare_words_threshold = kwargs.get("rare_words_threshold")
        logger.info(f"Nr. before removing words with at most {rare_words_threshold} appearances: {len(counts.keys())}")
        for word in list(counts):
            if counts[word] <= rare_words_threshold:
                del counts[word]
        logger.info(f"Nr. after removing words with at most {rare_words_threshold} appearances: {len(counts.keys())}")

    vocab2index = {"": 0, "UNK": 1}
    vocab = list(vocab2index.keys())
    for word in counts:
        vocab2index[word] = len(vocab)
        vocab.append(word)

    return vocab, vocab2index


def split_data(df: pd.DataFrame,
               splits: Tuple,
               test_size: float = 0.1,
               valid_size: float = 0.11,
               random_state: int = 21,
               *args: str,
               **kwargs: int) -> Dict[str, pd.DataFrame]:
    """
    Splits a dataframe into train/valid/test data and returns a dictionary where they keys are the names of the splits
    and the values are the dataframes.

    :param df: A pandas Dataframe representing the dataset.
    :param splits: A tuple containing the name of the splits (train/valid/test by default).
    :param test_size: The percent of test samples from the total.
    :param valid_size: The percent of valid samples from the remaining (total - test).
    :param random_state: A random state for reproductibility.
    :param args: Non-keyword varargs.
    :param kwargs: Keyword varargs.

    :return: A dictionary containing the name of splits and the corresponding dataframes.
    """
    logger.debug("Reading data...")

    # Source: https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df

    df = df[df['content'].map(lambda x: len(x.split())) <= 64]

    _train_df, test_df = train_test_split(df,
                                          test_size=test_size,
                                          random_state=random_state,
                                          stratify=df["label"])
    train_df, valid_df = train_test_split(_train_df,
                                          test_size=valid_size,
                                          random_state=random_state,
                                          stratify=_train_df["label"])

    return dict(zip(splits, [train_df, valid_df, test_df]))


def save_data(dfs: Dict[str, pd.DataFrame],
              paths: Tuple) -> None:
    """
    Saves the data locally in the specified paths.

    :param dfs: A dictionary where the keys are the name of the split and the values are corresponding dataframes.
    :param paths: A tuple containing the names of the split (train/valid/test by default).

    :return: None
    """
    for index, (name, df) in enumerate(dfs.items()):
        df.to_csv(paths[index], index=False)
        logging.debug(f"Saving {name} data")


def create_datasets(df: pd.DataFrame,
                    splits: Tuple,
                    paths: Tuple) -> dict:
    """
    A wrapper to create the datasets.

    :param df: An unsplit pandas DataFrame.
    :param splits: The names of the splits.
    :param paths: The paths where splits are saved.

    :return: None
    """
    dfs = split_data(df, splits)
    save_data(dfs, paths)
    return dfs
