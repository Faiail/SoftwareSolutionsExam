from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def split_dataset(data_path):
    dataset = pd.read_csv(data_path, index_col=0)
    train_data, test_data = train_test_split(dataset, stratify=dataset['style'],
                                             test_size=.2, shuffle=True,
                                             random_state=42)
    train_data.to_csv('data/train_data.csv')
    test_data.to_csv('data/test_data.csv')


if __name__ == '__main__':
    split_dataset('data/dataset/dataset.csv')