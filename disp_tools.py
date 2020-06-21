
from tabulate import tabulate


def pd_tab(df):
    return tabulate(df, headers='keys', tablefmt='psql')


def pd_samp(df, nrows=10):
    return tabulate(df[:nrows], headers='keys', tablefmt='psql')


def pd_desc(df):
    return tabulate(df.describe(include='all'),
                    headers='keys',
                    tablefmt='psql')
