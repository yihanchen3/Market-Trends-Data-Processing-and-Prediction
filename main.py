import datetime
import pandas as pd
from src.acquiring import *
from src.storing import *
from src.processing import *
from src.exploring import *
from src.inference import *


def main():
    ''' DATA ACQUISITION '''
    ''' acquire stock data '''
    COMPANY_DATA = 'AAL'
    START_DATE = datetime.datetime(2017, 4, 1)
    END_DATE = datetime.datetime(2022, 5, 31)
    stock_df = acquire_finance_data(
        COMPANY_DATA, START_DATE, END_DATE).loc[:, ['Close']]
    stock_df.columns = ['stock price']
    stock_df = stock_df.rename(
        index=lambda c: datetime.datetime.strptime(str(c.date()), '%Y-%m-%d'))
    # print('dataframe of %s from %s to %s \n' % (COMPANY_DATA,START_DATE,END_DATE), stock_df)

    ''' acquire covid data '''
    AUXILIARY_DATA = 'Covid Cases and Deaths'
    AUXILIARY_DATA_URL = 'https://github.com/nytimes/covid-19-data/blob/master/us.csv' + '?raw=true'
    START_DATE = '2017-04-01'
    END_DATE = '2022-05-31'
    covid_df = acquire_auxiliary_data(AUXILIARY_DATA_URL, START_DATE, END_DATE)
    covid_df.columns = ['covid_cases', 'covid_deaths']
    covid_df = covid_df.rename(
        index=lambda c: datetime.datetime.strptime(c, '%Y-%m-%d'))
    # print('dataframe of %s from %s to %s \n' % (AUXILIARY_DATA,START_DATE,END_DATE), covid_df)

    ''' DATA STORGE '''
    ''' store data cloud-based and locally'''
    store_cloud(stock_df, covid_df)
    store_local(stock_df, covid_df, COMPANY_DATA, AUXILIARY_DATA)

    ''' split stock dataset from May 2022 for the final forecasting '''
    predict_df = stock_df[datetime.date(2022, 5, 1):]
    stock_df = stock_df[:datetime.date(2022, 4, 30)]
    data_plot(stock_df, COMPANY_DATA, './img/stock_original.png')

    ''' DATA PROCESSING '''
    ''' covid data processing with differetial calculation '''
    data_plot(covid_df.loc[:, ['covid_cases', 'covid_deaths']],
              AUXILIARY_DATA, './img/covid_original.png')
    covid_df['cases_diff'] = calculate_diff(covid_df, 'covid_cases')
    covid_df['deaths_diff'] = calculate_diff(covid_df, 'covid_deaths')
    predict_covid_df = covid_df[datetime.date(2022, 5, 1):]
    covid_df = covid_df[:datetime.date(2022, 4, 30)]
    data_plot(covid_df.loc[:, ['cases_diff', 'deaths_diff']],
              AUXILIARY_DATA, './img/covid_diff.png')

    ''' stock and covid data procesiing with normalization '''
    stock_df_concat = df_concat(
        stock_df, covid_df.loc[:, ['cases_diff', 'deaths_diff']])
    df_norm = df_normalize(stock_df)
    df_norm_covid = df_normalize(stock_df_concat)
    data_plot(df_norm_covid, 'Normalization', './img/df_normalization.png')
    plt.close('all')

    ''' DATA EXPLORATION '''
    ''' seasonality analyse'''
    seasonal_explore(stock_df, './img/seasonal_explore.png')
    month_heatmap(stock_df, './img/month_heatmap.png')

    ''' stock and covid data correlation analyse'''
    # histogram(stock_df)
    # stock_statistics_dict = statistics(stock_df['stock price'].values)
    # boxplot((stock_statistics_dict['min']+stock_statistics_dict['max'])/2,stock_statistics_dict['min'],stock_statistics_dict['max'], stock_statistics_dict['percentile_25'],stock_statistics_dict['percentile_75'])
    stock_covid_df = stock_df_concat[datetime.datetime(2020, 1, 21):]
    correlation(stock_covid_df, 'stock price', 'cases_diff', './img/')
    correlation(stock_covid_df, 'stock price', 'deaths_diff', './img/')
    plt.close('all')

    ''' DATA INFERENCE '''
    ''' prophet model with stock data'''
    df_stock_input = data_input(df_norm, target_feature='stock price')
    train, test = train_test_split(df_stock_input, '2021-03-31', '2021-04-01')
    model_prophet = model(yearly=True, weekly=False, daily=False)
    model_prophet.fit(train)
    future = df_stock_input[['ds']]
    forecast = model_prophet.predict(future)
    f = model_prophet.plot_components(forecast, figsize=(12, 16))
    result = result_visualising(forecast, train, test, 'stock')
    # create_joint_plot(result.loc[:'2020-03-31', :], 'stock', title='Train set')
    # create_joint_plot(result.loc['2020-04-01':, :], 'stock', title='Test set')
    plt.close('all')

    ''' prophet model with stock and covid data'''
    df_stock_covid_input = data_input(
        df_norm_covid, target_feature='stock price')
    train_covid, test_covid = train_test_split(
        df_stock_covid_input, '2021-03-31', '2021-04-01')
    model_prophet_covid = model(yearly=True, weekly=False, daily=False)
    model_prophet_covid.add_regressor('cases_diff', mode='multiplicative')
    model_prophet_covid.add_regressor('deaths_diff', mode='multiplicative')
    model_prophet_covid.fit(train_covid)
    futures_covid = pd.concat([future, df_norm_covid.loc[:, [
                              'cases_diff', 'deaths_diff']].reset_index(drop=True)], axis=1)
    forecast_covid = model_prophet_covid.predict(futures_covid)
    f_covid = model_prophet_covid.plot_components(
        forecast_covid, figsize=(12, 16))
    result_covid = result_visualising(
        forecast_covid, train_covid, test_covid, 'covid')
    # create_joint_plot(result_covid.loc[:'2020-03-31', :], 'covid', title='Train set')
    # create_joint_plot(result_covid.loc['2020-04-01':, :], 'covid', title='Test set')
    plt.close('all')

    ''' evaluate model with May 2022 data '''
    '''' model with stock data '''
    predict_df_norm = df_normalize(predict_df)
    df_May_input = data_input(predict_df_norm, target_feature='stock price')
    future_May = df_May_input[['ds']]
    forecast_May = model_prophet.predict(future_May)
    mae1, mse1, r21 = evaluation_metrics(df_May_input, forecast_May)
    ''' model with stock and covid data '''
    stock_df_concat_May = df_concat(
        predict_df, predict_covid_df.loc[:, ['cases_diff', 'deaths_diff']])
    df_norm_covid_May = df_normalize(stock_df_concat_May)
    futures_May_covid = pd.concat([future_May, df_norm_covid_May.loc[:, [
                                  'cases_diff', 'deaths_diff']].reset_index(drop=True)], axis=1)
    forecast_May_covid = model_prophet_covid.predict(futures_May_covid)
    mae2, mse2, r22 = evaluation_metrics(df_May_input, forecast_May_covid)

    print('Evaluation metrics of model with stock data: MAE: %.4f, MSE: %.4f, R2: %.4f' % (
        mae1, mse1, r21))
    print('Evaluation metrics of model with stock and covid data: MAE: %.4f, MSE: %.4f, R2: %.4f' % (
        mae2, mse2, r22))
    May_plot(df_May_input, forecast_May, forecast_May_covid)
    plt.close('all')


if __name__ == "__main__":
    main()
    print('program end')
