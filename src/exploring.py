import pickle
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from calendar import day_abbr, month_abbr
from scipy.stats import pearsonr, spearmanr
import datetime


def seasonal_explore(df, save_path):
    seasonal_df = df.rolling(window=30, center=True).mean().groupby(df.index.dayofyear).mean()
    q25 = df.rolling(window=30, center=True).mean().groupby(df.index.dayofyear).quantile(0.25)
    q75 = df.rolling(window=30, center=True).mean().groupby(df.index.dayofyear).quantile(0.75)

    month_ticks = month_abbr[1:]
    fig, ax = plt.subplots(figsize=(16, 7))

    seasonal_df.plot(ax=ax, lw=2, color='b', legend=False)
    ax.fill_between(seasonal_df.index, q25.values.ravel(), q75.values.ravel(), color='b', alpha=0.3)
    ax.set_xticklabels(month_ticks)
    ax.grid(ls=':')
    ax.set_xlabel('Month', fontsize=15)
    ax.set_ylabel('stock prices', fontsize=15)
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

    ax.set_title('30 days running average stock prices', fontsize=15)
    plt.savefig(save_path)


def month_heatmap(data, save_path):
    month_year = data.copy()
    month_year.loc[:, 'year'] = month_year.index.year
    month_year.loc[:, 'month'] = month_year.index.month
    month_year = month_year.groupby(['year', 'month']).mean().unstack()
    month_year.columns = month_year.columns.droplevel(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    sn.heatmap(month_year, ax=ax, cmap=plt.cm.viridis, cbar_kws={'boundaries': np.arange(10, 70, 10)})

    cbax = fig.axes[1]
    [l.set_fontsize(13) for l in cbax.yaxis.get_ticklabels()]
    cbax.set_ylabel('stock prices', fontsize=13)

    [ax.axhline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 7)]
    [ax.axvline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 24)]

    ax.set_title('stock prices per year and month', fontsize=16)

    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

    ax.set_xlabel('Month', fontsize=15)
    ax.set_ylabel('Year', fontsize=15)
    ax.set_yticklabels(np.arange(2017, 2022 + 1, 1), rotation=0)
    plt.savefig(save_path)


def statistics(distribution):
    array = np.array(distribution)
    mean = np.mean(array)
    std = np.std(array)
    minimum = np.min(array)
    maximum = np.max(array)
    percentile_5 = np.percentile(array, 5)
    percentile_10 = np.percentile(array, 10)
    percentile_25 = np.percentile(array, 25)
    percentile_75 = np.percentile(array, 75)
    percentile_90 = np.percentile(array, 90)
    percentile_95 = np.percentile(array, 95)

    return {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "percentile_5": percentile_5,
        "percentile_10": percentile_10,
        "percentile_25": percentile_25,
        "percentile_75": percentile_75,
        "percentile_90": percentile_90,
        "percentile_95": percentile_95,
    }


def correlation(df, col1_name, col2_name, save_path):
    x = df[col1_name].values
    y = df[col2_name].values

    fig = plt.figure()
    df.plot.scatter(x=col1_name, y=col2_name)
    plt.title('scatter of %s and %s' % (col1_name, col2_name))
    plt.savefig(save_path + 'scatter of %s and %s' % (col1_name, col2_name) + '.png')

    cov_xy = np.cov(x, y)
    fig = plt.figure()
    sn.heatmap(cov_xy, annot=True, fmt='g')
    plt.title('covariance matrix of %s and %s' % (col1_name, col2_name))
    plt.savefig(save_path + 'covariance matrix of %s and %s' % (col1_name, col2_name) + '.png')

    corr, p = spearmanr(x, y)
    print('%s and %s correlation : %f p_value: %f' % (col1_name, col2_name, corr, p))

def histogram(data):
    fig, axes = plt.subplots()
    axes.hist(data)
    axes.set_title('distribution of stock prices data')
    axes.set_xlabel("number of each stock prices range")
    axes.set_ylabel("stock prices")
    plt.savefig('./img/histogram.png')
    return fig, axes


def boxplot(median, minimum, maximum, quantile_1, quantile_3):
    fig, axes = plt.subplots()
    boxes = [
        {
            "label": "Aggregated statistics for stock prices",
            "whislo": minimum,
            "q1": quantile_1,
            "med": median,
            "q3": quantile_3,
            "whishi": maximum,
            "fliers": [],
        }
    ]
    axes.bxp(boxes, showfliers=False)
    axes.set_ylabel("stock prices")
    plt.savefig('./img/boxplot.png')
    return fig, axes