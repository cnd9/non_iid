import scipy
from itertools import chain, repeat

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

class Bootstrap:
    def __init__(self):
        self.bootstrap_samples = None

    def iid_bootstrap(self, df, b, dcol='value', use_poisson=True, func=lambda x: x):

        boots = []
        values = df[dcol].values
        if not use_poisson:
            weights = scipy.random.multinomial(len(df), [1 / len(df)] * len(df), size=b)
        else:
            weights = scipy.random.poisson(1, size=(b, len(df)))
        for weight in weights:
            boots.append(list(chain.from_iterable(map(repeat, values, weight))))
        self.bootstrap_samples = boots
        return [func(bs) for bs in boots]

    def block_bootstrap(self, df, b, n_dims=1, dcol='value', icol=['uid'], func=lambda x: x):
        if len(icol) != n_dims:
            raise Exception('Cluster index indicators must have length of n_dims')
        if any(i not in df.columns for i in icol + [dcol]):
            raise Exception('Input dataframe must contain indicated data column and index column(s)')
        boots = []
        group_df = df.groupby(icol).agg({dcol: lambda x: list(x)}).reset_index()
        values = group_df[dcol].values
        weights = scipy.random.poisson(1, size=(b, len(group_df)))
        for weight in weights:
            boots.append(flatten(list(chain.from_iterable(map(repeat, values, weight)))))
        self.bootstrap_samples = boots
        return [func(bs) for bs in boots]