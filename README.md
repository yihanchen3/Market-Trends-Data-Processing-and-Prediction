# Stock Data Processing and Forecasting Pipeline

This project aims to build a complete pipeline to acquire data of selected companies, study their market trends witha auxiliary data, and ultimately forecast stock prices.
The pipeline includes data acquisition, storage, processing, exploration, and inference using the Prophet forecasting model.
This project selected American Airlines Group Inc (AAL) as the source of financial data to investigate into and Covid-19 statistics to integrate with.

## Overview

- **Acquiring Data:**
  - Stock data for a specified company ('AAL') and Covid-19 cases and deaths data are acquired from relevant sources.
  - Data is processed and stored both locally and on cloud services.

- **Data Processing:**
  - Stock data is split into historical and forecast periods for training and testing.
  - Covid data is differentially calculated and normalized for improved model performance.

- **Data Exploration:**
  - Seasonality analysis of stock data.
  - Correlation analysis between stock prices and Covid data.

- **Data Inference:**
  - Two forecasting models are implemented using the Prophet model:
    - Model with stock data only.
    - Model with both stock and Covid data.

## How to run this project

- Use the `environment.yml` file to reconstruct the environment of the  project.
- Run the `main.py` file to execute the entire pipeline, including data acquisition, storage, processing, exploration, and inference.
- Evaluation metrics are provided for forecasting accuracy with a focus on May 2022 data.

## Folder Structure

- `src/`: Contains modules for data acquisition, storage, processing, exploration, and inference.
- `img/`: Stores generated images from the data exploration phase, and Results of the forecasting models, including visualizations and evaluation metrics.

## Role of each file

- **main.py** calls the functions from other supplementary files and performs the data acquisition, storage, processing, exploring and infeerence in order. It is responsible to train and test models for stock prices forecasting task.
- **acquiring.py** defines the functions to acquire the stock and covid datasets from the data sources.
- **storing.py** defines the functions to storage data to cloud-based database and locally.
- **processing.py** defines the functions to clean, visualize and transform raw data including normalization and differencial calculation.
- **exploring.py** defines the EDA methods to analyse the seaonality and dependency of data, as well as calculating the correlation between stock data and covid data.
- **inference.py** defines the facebook prophet model and the result visualization and the evaluation metrics calculations.
- **src folder** contains the support files to run the main file.
- **img folder** contains the graphs generated during program running.
- **data folder** contains the datasets for acquired data storage.
