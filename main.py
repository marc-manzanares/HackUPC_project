import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import register_matplotlib_converters
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import aic, bic, hqic
import sys

current_path = os.getcwd()
dataset_path=current_path + "/dataset/"

from sklearn.preprocessing import StandardScaler

def clean_dataset_test(dataframe):
    # Drop missing values
    dataframe = dataframe.dropna()

    # Drop duplicates
    dataframe = dataframe.drop_duplicates()

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    dataframe[['inventory_units', 'sales_units']] = scaler.fit_transform(dataframe[['inventory_units', 'sales_units']])
    dataframe.rename(columns={'inventory_units': 'normalized_inventory_units', 'sales_units': 'normalized_sales_units'}, inplace=True)

    # Set date as the index
    dataframe.set_index('date', inplace=True)
    print(dataframe.index.is_monotonic_increasing)

    return dataframe


def clean_dataset(dataframe):
    # Drop missing values
    dataframe = dataframe.dropna()

    # Drop duplicates
    dataframe = dataframe.drop_duplicates(['id', 'year_week', 'product_number'])

    # Convert date to datetime type
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe = dataframe.sort_values('date')

    # Fill missing values for sales_units column
    dataframe['sales_units'] = pd.to_numeric(dataframe['sales_units'], errors='coerce')
    dataframe['sales_units'].fillna(method='ffill', inplace=True)

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    dataframe[['inventory_units', 'sales_units']] = scaler.fit_transform(dataframe[['inventory_units', 'sales_units']])
    dataframe.rename(columns={'inventory_units': 'normalized_inventory_units', 'sales_units': 'normalized_sales_units'}, inplace=True)

    # Set date as the index
    dataframe.set_index('date', inplace=True)
    print(dataframe.index.is_monotonic_increasing)

    return dataframe

def check_stationary(df):
   # Perform the ADF test on the differenced time series
    result = adfuller(df)

    # Print the test statistics and p-value
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')

    if isinstance(df, pd.DataFrame):
        print(df.columns[0])
    elif isinstance(df, pd.Series):
        print(df.name)
    else:
        raise ValueError("Input must be a Pandas DataFrame or Series")

def autocorrelation_function(df):
    plot_acf(df)
    plt.show()

def partial_autocorrelation_function(df):
    # Set the 'date' column as the index
    df.set_index('date', inplace=True)

    # Plot the PACF for the 'normalized_inventory_units' column
    plot_pacf(df)
    plt.show()

    # Plot the PACF for the 'normalized_sales_units' column
    plot_pacf(df)
    plt.show()
    
def main():
    # Clean dataset and normalized
    training_dataset = dataset_path + 'train.csv'
    test_dataset = dataset_path + 'test.csv'

    df = clean_dataset(pd.read_csv(training_dataset))
    df.index = pd.DatetimeIndex(df.index).to_period('M')

    #df_test = clean_dataset(pd.read_csv(test_dataset))

    train_size = int(len(df) * 0.8)
    train_df, val_df = df[:train_size], df[train_size:]

    # Showing that the dataset is non-stationary
    check_stationary(train_df['normalized_inventory_units'])
    check_stationary(train_df['normalized_sales_units'])
    
    # Load and preprocess the training dataset to make it stationary
    train_df = train_df[['normalized_inventory_units', 'normalized_sales_units']]
    train_df = train_df.diff().dropna()

    # Build the VAR model with lag 1
    model = VAR(train_df)
    results = model.fit(1)

    # Read test data
    test_df = pd.read_csv(test_dataset)

    # Get unique ids from the test data
    ids = test_df['id'].unique()

    # Initialize empty dataframes for the predictions
    predictions_inventory = pd.DataFrame(columns=['id', 'date', 'inventory_units'])
    predictions_sales = pd.DataFrame(columns=['id', 'date', 'sales_units'])

    # Iterate over ids and make predictions
    for id in ids:
        # Filter training data for this id

        print(train_df.columns)

        train_df_id = df.loc[df['id'] == id]

        # Convert training data to stationary time series
        train_df_id = train_df_id[['normalized_inventory_units', 'normalized_sales_units']]
        train_df_id = train_df_id.diff().dropna()

        # Make predictions using VAR model
        predictions = results.forecast(train_df_id.values[-1:], len(test_df.loc[test_df['id'] == id]))

        # Format predictions and add to result dataframes
        predictions_inventory_id = pd.DataFrame(predictions[:, 0], columns=['inventory_units'])
        predictions_sales_id = pd.DataFrame(predictions[:, 1], columns=['sales_units'])
        predictions_inventory_id['id'] = id
        predictions_sales_id['id'] = id
        predictions_inventory_id['date'] = test_df.loc[test_df['id'] == id]['date'].values
        predictions_sales_id['date'] = test_df.loc[test_df['id'] == id]['date'].values
        predictions_inventory = pd.concat([predictions_inventory, predictions_inventory_id])
        predictions_sales = pd.concat([predictions_sales, predictions_sales_id])

    # Save predictions to CSV files
    predictions_inventory.to_csv('predictions_inventory.csv', index=False)
    predictions_sales.to_csv('predictions_sales.csv', index=False)

"""
    df_seasonal_diff_inventory = df_stationary_inventory.diff(periods=1).dropna()
    merged_df_seasonal_diff_inventory = dates.merge(df_seasonal_diff_inventory, left_index=True, right_index=True)
    df_seasonal_diff_sales = df_stationary_sales.diff(periods=1).dropna()
    merged_df_seasonal_diff_sales = dates.merge(df_seasonal_diff_sales, left_index=True, right_index=True)

    merged_inventory_sales = df_seasonal_diff_inventory.merge(df_seasonal_diff_sales, left_index=True, right_index=True)
    merged_all = dates.merge(merged_inventory_sales, left_index=True, right_index=True)

    print("Autocorrelation with seasonal diff stationary inventory and stationary sales")
    autocorrelation_function(merged_df_seasonal_diff_inventory)
    autocorrelation_function(merged_df_seasonal_diff_sales)

    # Fit VAR model with maximum lag of 10
    model = VAR(merged_all)
    results = model.fit(maxlags=10)

    # Print information criterion for different lag lengths
    for lag in range(1, 11):
        print(f"Lag {lag}")
        print(f"AIC: {aic(results, lag)}")
        print(f"BIC: {bic(results, lag)}")
        print(f"HQIC: {hqic(results, lag)}")
"""


if __name__ == "__main__":
    main()