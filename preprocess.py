import pandas as pd
from data_helpers import load_data


def preprocessing(df):
    # remove difficult to parse columns
    df = df.drop(['Name', 'Ticket'], axis=1)

    # replace missing values with median
    cols = ['Age', 'SibSp', 'Parch', 'Fare']
    for col in cols:
        df[col] = df[[col]].fillna(df[[col]].median())

    # replace missing values with 'Unknown'
    cols = ['Cabin', 'Embarked']
    for col in cols:
        df[col] = df[[col]].fillna('U')

    # for cabin, only consider letter:
    df.Cabin = df.Cabin.apply(lambda x: x[0])

    # one-hot-encode text columns, using all categories appearing
    # in test or train sets
    df.Sex = df.Sex.astype(pd.CategoricalDtype(
        categories=['male', 'female']
    ))
    df.Embarked = df.Embarked.astype(pd.CategoricalDtype(
        categories=['Q', 'S', 'C', 'U']
    ))
    df.Cabin = df.Cabin.astype(pd.CategoricalDtype(
        categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U']
    ))

    cols = ['Sex', 'Cabin', 'Embarked']
    for col in cols:
        df = pd.concat([df, pd.get_dummies(df[[col]])], axis=1)
        df = df.drop([col], axis=1)
    return df


if __name__ == '__main__':
    train_df = load_data('train', preprocessed=False)
    train_df = preprocessing(train_df)
    train_df.to_csv('preprocessed_data/train_data/train.csv')

    test_df = load_data('test', preprocessed=False)
    test_df = preprocessing(test_df)
    test_df.to_csv('preprocessed_data/test_data/test.csv')
