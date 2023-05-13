import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import register_matplotlib_converters
import numpy as np
from statsmodels.tsa.stattools import adfuller

current_path = os.getcwd()
dataset_path=current_path + "/dataset/"

def clean_dataset(dataframe):
    # Drop missing values
    dataframe = dataframe.dropna()
    # Drop duplicates
    dataframe = dataframe.drop_duplicates(['id', 'year_week', 'product_number'])

    # it is useful to set the correct type for the date for some plotting functions
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['sales_units'] = pd.to_numeric(dataframe['sales_units'])

    # Check for errors and inconsistencies
    # and perform necessary cleaning steps

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    dataframe['normalized_sales_units'] = scaler.fit_transform(dataframe[['sales_units']])
    dataframe['normalized_inventory_units'] = scaler.fit_transform(dataframe[['inventory_units']])

    dataframe = dataframe.drop(columns=['sales_units', 'inventory_units'])

    # Return the cleaned dataset
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
    
    if result[1] == 0 and result[0] < 0:
        print("Is stationary")
    else:
        print("Is not stationary")

    print("\n")

def autocorrelation_function(df):
    # Set the 'date' column as the index
    df.set_index('date', inplace=True)

    # Plot the ACF for the 'inventory_units' column
    plot_acf(df)
    plt.show()

    # Plot the ACF for the 'sales_units' column
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

    df = clean_dataset(pd.read_csv(training_dataset))
    
    # Showing that the dataset is non-stationary
    check_stationary(df['normalized_inventory_units'])
    check_stationary(df['normalized_sales_units'])

    # Take the first difference of the time series (making it stationary)
    df_stationary_inventory = df[['normalized_inventory_units']].diff().dropna()
    df_stationary_sales = df[['normalized_sales_units']].diff().dropna()
    
    # Proving it is now stationary
    check_stationary(df_stationary_inventory)
    check_stationary(df_stationary_sales)

    dates = df[['date']]
    merged_df_stationary_inventory = dates.merge(df_stationary_inventory, left_index=True, right_index=True)
    merged_df_stationary_sales = dates.merge(df_stationary_sales, left_index=True, right_index=True)

    # Calculate the full and the partial autocorrelation function
    autocorrelation_function(merged_df_stationary_inventory)
    autocorrelation_function(merged_df_stationary_sales)

    # TODO: No entenc perque se'm queixa
    partial_autocorrelation_function(merged_df_stationary_inventory)
    partial_autocorrelation_function(merged_df_stationary_sales)

if __name__ == "__main__":
    main()