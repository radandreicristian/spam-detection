import logging
import os
from collections import Counter
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def create_vocabulary(df,
                      *args,
                      **kwargs):
    counts = Counter()
    for index, row in df.iterrows():
        content_words = [token for token in row['content'].split()]
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


def split_data(df,
               splits,
               test_size=0.1,
               valid_size=0.11,
               random_state=21,
               *args,
               **kwargs) -> Dict[str, pd.DataFrame]:
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
              *paths) -> None:
    for index, (name, df) in enumerate(dfs.items()):
        df.to_csv(paths[index], index=False)
        logging.debug(f"Saving {name} data")


def create_datasets(df, splits, *paths):
    dfs = split_data(df, splits)
    save_data(dfs, *paths)
