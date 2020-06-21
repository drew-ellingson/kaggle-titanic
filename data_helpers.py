import pandas as pd


def load_data(dataset, preprocessed=True):
    file_prefix = 'preprocessed_data' if preprocessed else 'raw_data'
    if dataset == 'train':
        df = pd.read_csv(file_prefix+'/train_data/train.csv', index_col=0)
    elif dataset == 'test':
        df = pd.read_csv(file_prefix+'/test_data/test.csv', index_col=0)
    else:
        raise ValueError("Please enter 'test' or 'train' value for dataset \
                         argument")
    return df


def transform_train():
    df = load_data('train')
    y = df[['Survived']]
    X = df.drop('Survived', axis=1)
    return X, y
