from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def model(yearly=True, weekly=True, daily=False):
    m = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily)
    return m


def data_input(data, target_feature):
    new_data = data.copy()
    new_data.reset_index(inplace=True)
    new_data = new_data.rename({'Date': 'ds', '{}'.format(target_feature): 'y'}, axis=1)
    return new_data


def train_test_split(data, split_date1, split_date2):
    train = data.set_index('ds').loc[:split_date1, :].reset_index()
    test = data.set_index('ds').loc[split_date2:, :].reset_index()

    return train, test


def make_predictions_df(forecast, data_train, data_test):
    """
    Function to convert the output Prophet dataframe to a datetime index and append the actual target values at the end
    """
    forecast.index = pd.to_datetime(forecast.ds)
    data_train.index = pd.to_datetime(data_train.ds)
    data_test.index = pd.to_datetime(data_test.ds)
    data = pd.concat([data_train, data_test], axis=0)
    forecast.loc[:, 'y'] = data.loc[:, 'y']

    return forecast


def plot_predictions(forecast, start_date):
    """
    Function to plot the predictions
    """
    f, ax = plt.subplots(figsize=(14, 8))

    train = forecast.loc[start_date:'2021-03-31', :]
    ax.plot(train.index, train.y, 'ko', markersize=3)
    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)

    test = forecast.loc['2021-04-01':, :]
    ax.plot(test.index, test.y, 'ro', markersize=3)
    ax.plot(test.index, test.yhat, color='coral', lw=0.5)
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)
    ax.axvline(forecast.loc['2021-04-01', 'ds'], color='k', ls='--', alpha=0.7)

    ax.grid(ls=':', lw=0.5)

    return f, ax


def result_visualising(forecast, train, test, data_name):
    result = make_predictions_df(forecast, train, test)
    result.loc[:, 'yhat'] = result.yhat.clip(lower=0)
    result.loc[:, 'yhat_lower'] = result.yhat_lower.clip(lower=0)
    result.loc[:, 'yhat_upper'] = result.yhat_upper.clip(lower=0)
    result.head()
    f, ax = plot_predictions(result, '2017-04-01')
    plt.savefig('./img/result_hat_' + data_name + '.png')
    return result


def create_joint_plot(forecast, data_name, x='yhat', y='y', title=None):
    g = sns.jointplot(x='yhat', y='y', data=forecast, kind="reg", color="b")
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    ax = g.fig.axes[1]
    if title is not None:
        ax.set_title(title, fontsize=16)

    ax = g.fig.axes[0]
    ax.text(5, 50, "R = {:+4.2f}".format(forecast.loc[:, ['y', 'yhat']].corr().iloc[0, 1]), fontsize=16)
    ax.set_xlabel('Predictions', fontsize=15)
    ax.set_ylabel('Observations', fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(ls=':')
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

    ax.grid(ls=':')
    plt.savefig('./img/result_joint_plot_' + data_name + '.png')


def mse_f(y_test, y_predict):
    return np.sum((y_test - y_predict) ** 2) / len(y_test)


def mae_f(y_test, y_predict):
    return np.sum(np.absolute(y_test - y_predict)) / len(y_test)


def r2_f(y_test, mse):
    return 1-mse/ np.var(y_test)


def evaluation_metrics(df_test, df_predict):
    y_test = df_test['y'].values
    y_predict = df_predict ['yhat'].values
    mae = mae_f(y_test,y_predict)
    mse = mse_f(y_test,y_predict)
    r2 = r2_f(y_test,mse)
    return mae, mse, r2


def May_plot(df_test, df_predict_stock, df_predict_stock_covid):
    fig, ax = plt.subplots()
    x = np.linspace(0,1,20)
    ax.plot(x, df_test['y'].values, label='original')
    ax.plot(x, df_predict_stock['yhat'].values, label='predict with stock')
    ax.plot(x, df_predict_stock_covid['yhat'].values, label='predict with stock and covid')
    ax.set_xlabel('May date')
    ax.set_ylabel('stock price')
    ax.set_title('May predict plot')
    ax.legend()
    plt.savefig('./img/May_predict_plot.png')