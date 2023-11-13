"""
This module contains methods to perform numerical processing of the acquired data.
"""

import matplotlib.pyplot as plt
import numpy as np


def calculate_diff(df, col_name):
    Q1 = np.array(df[col_name].values).T
    Q2 = df[col_name].tolist()
    Q2.insert(0, 0)
    Q2.pop()
    Q2 = np.array(Q2).T
    Q3 = Q1 - Q2
    return Q3.tolist()


def data_plot(data, data_name, save_path):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(data)
    ax.grid(ls='--', c='k', alpha=0.2)
    ax.set_xlabel('date')
    ax.set_ylabel(data_name)
    plt.title(save_path)
    plt.savefig(save_path)


def df_concat(df_origin, df_external):
    df = df_origin.copy()
    for i in df_external.columns:
        df[i] = df_external[i]
        df[i].fillna(0, inplace=True)
    return df


def df_normalize(df):
    return df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
