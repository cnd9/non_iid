import scipy
import pandas as pd
import numpy as np
from itertools import chain, repeat
from .utils import *

class Bootstrap:
    def __init__(self):
        self.bootstrap_samples = None

    def __init__(self):
        self.bootstrap_samples = None

    def iid_bootstrap(self, df, b, dcol='value', use_poisson=True, func=lambda x: x):

        """'counts' is a len(df) * b (number of bootstrap samples) array
        indicating how many times a given datapoint d_i appears in the b_jth bootstrap sample.
        In the exact case (using the multinomial distribution), each bootstrap sample is
        constrained to have exactly len(df) datapoints.
        See https://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html
        for the poisson approximation"""

        boots = []
        values = df[dcol].values
        if not use_poisson:
            counts = scipy.random.multinomial(len(df), [1 / len(df)] * len(df), size=b)
        else:
            counts = scipy.random.poisson(1, size=(b, len(df)))
        for count in counts:
            boots.append(list(chain.from_iterable(map(repeat, values, count))))
        self.bootstrap_samples = boots
        return [func(bs) for bs in boots]

    def block_bootstrap(self, df, b, dcol='value', icols=['uid'], func=lambda x: x):

        """n-way block bootstrap.  """
        uniques = {}
        weight_cols = [f'w_{icol}' for icol in icols]
        if any(i not in df.columns for i in icols + [dcol]):
            raise Exception('Input dataframe must contain indicated data column and index column(s)')
        boots = []
        group_df = df.groupby(icols).agg({dcol: lambda x: list(x)}).reset_index()
        for icol in icols:
            uniques[icol] = list(group_df[icol].unique())
        for b_j in range(b):
            for icol in icols:
                counts = scipy.random.poisson(1, size=(len(uniques.get(icol))))
                mdf = pd.DataFrame(columns=[icol, f'w_{icol}'],
                                   data=np.array([uniques.get(icol), counts]).transpose())
                group_df = group_df.merge(mdf, on=icol)
            counts = np.prod(group_df[weight_cols].values, axis=1)
            values = group_df[dcol].values
            group_df = group_df[icols + [dcol]]
            boots.append(flatten(list(chain.from_iterable(map(repeat, values, counts)))))
        self.bootstrap_samples = boots
        return [func(bs) for bs in boots]