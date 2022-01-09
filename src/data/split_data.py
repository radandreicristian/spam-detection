import logging
import os
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def split_data(path: str,
               test_size=0.1,
               valid_size=0.11,
               random_state=21) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path, index_col=0)
    _train_df, test_df = train_test_split(df,
                                          test_size=test_size,
                                          random_state=random_state,
                                          stratify=df["label"])
    train_df, valid_df = train_test_split(_train_df,
                                          test_size=valid_size,
                                          random_state=random_state,
                                          stratify=_train_df["label"])

    return {'train': train_df,
            'valid': valid_df,
            'test': test_df}


def save_data(dfs: Dict[str, pd.DataFrame],
              path: str = 'data/processed') -> None:
    for name, df in dfs.items():
        complete_path = os.path.join(path, name, 'data.csv')
        df.to_csv(complete_path, index=False)
        logging.debug(f"Saving {name} data")


if __name__ == '__main__':
    PROCESSED_DATA_FILE_PATH = 'data/processed/preprocessed_data.csv'
    dfs = split_data(PROCESSED_DATA_FILE_PATH)
    save_data(dfs)
