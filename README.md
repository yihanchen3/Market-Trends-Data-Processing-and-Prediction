[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=9509284&assignment_repo_type=AssignmentRepo)

# Data Acquisition and Processing Systems (DAPS) (ELEC0136) -SN22049064

## How to run this project

To run the code, please use the command with `python main.py` .

## Environment required

Use the `environment.yml` file to reconstruct the environment of the  project.

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
