"""
A module to perform requests to the GitHub API.
It contains methods to construct, perform, and post-process requests.
"""

import yfinance as yahooFinance
import datetime
import pandas as pd
import numpy as np
import pymongo
import requests
import matplotlib.pyplot as plt
import seaborn as sns


def acquire_finance_data(company, start_date, end_date):
    GetFacebookInformation = yahooFinance.Ticker(company)
    # pass the parameters as the taken dates for start and end
    return GetFacebookInformation.history(start=start_date, end=end_date)


def acquire_auxiliary_data(url, start_date, end_date):
    covid_df = pd.read_csv(url, index_col=0)
    return covid_df.loc[start_date:end_date, :]

        # covid_df.loc[(covid_df.index <= end_date) & (covid_df.index >= start_date)]
