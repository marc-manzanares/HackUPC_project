import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import aic, bic, hqic
import sys
from pandas.plotting import autocorrelation_plot
import seaborn as sns
from statsmodels.tsa.stattools import kpss

current_path = os.getcwd()
dataset_path=current_path + "/dataset/"

from sklearn.preprocessing import StandardScaler

# our very simple ultra small data prep function
def prepare_df(df):
    df = df[['year_week', 'product_number', 'prod_category', 'segment']]
    df = pd.get_dummies(df, columns=['product_number', 'segment', 'prod_category'])
    return df

def clean_dataset_test(dataframe):
    # Drop missing values
    dataframe = dataframe.dropna()

    # Drop duplicates
    dataframe = dataframe.drop_duplicates()
    dataframe['id'] = dataframe['id'].str.replace('-', '').astype(int)
    dataframe = dataframe.sort_values(by=['id'])

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
    dataframe['id'] = dataframe['id'].str.replace('-', '').astype(int)

    # Set date as the index
    dataframe.set_index('date', inplace=True)
    print(dataframe.index.is_monotonic_increasing)

    return dataframe

def check_stationary(df):
    result = kpss(df)
    print('KPSS Statistic: {:.3f}'.format(result[0]))
    print('p-value: {:.3f}'.format(result[1]))
    print('Lags Used: {}'.format(result[2]))
    print('Critical Values:')
    for key, value in result[3].items():
        print('\t{}: {:.3f}'.format(key, value))

def autocorrelation_function(df):
   # Get the current x-axis limits
    acf_plot = plot_acf(df)
    x_limits = acf_plot.axes[0].get_xlim()

    # Set the x-axis limits to a new range of values
    acf_plot.axes[0].set_xlim([0, 4])
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
    # New code again

    # TRAIN DATA
    # Clean dataset and normalized
    training_dataset = dataset_path + 'train.csv'

    train_df = clean_dataset(pd.read_csv(training_dataset))

    gd = train_df.groupby('date').sum(numeric_only = True)
    plt.plot(gd.index, gd.inventory_units)
    plt.xticks(rotation = 45)
    plt.xlabel('Date') 
    plt.ylabel('Inventory Units')
    plt.title('Inventory in time')
    plt.show()

    plt.plot(gd.index, gd.sales_units)
    plt.xticks(rotation = 45)
    plt.xlabel('Date') 
    plt.ylabel('Sales Units')
    plt.title('Sales in time')
    plt.show()

    check_stationary(train_df['inventory_units'])
    check_stationary(train_df['sales_units'])

    # Load and preprocess the training dataset to make it stationary
    new = train_df[['inventory_units', 'sales_units']]
    new = new.diff().dropna()

    check_stationary(new['inventory_units'])
    check_stationary(new['sales_units'])

    autocorrelation_function(new['inventory_units'])
    autocorrelation_function(new['sales_units'])
    
    #LAG OF 1
    final = new[['inventory_units', 'sales_units']].diff(1)
    final = final.dropna()
    print(final)

    # Remove any rows with missing values (since the first row will now be missing)
    train_df = train_df.dropna()

    # it's interesting to know the correlation among variables
    # Calculate the correlation matrix for the numeric columns
    corr_matrix = train_df.corr(numeric_only=True)

    # Plot the correlation matrix as a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")

    # Display the plot
    plt.show()

    X_train = prepare_df(train_df)
    y_train = train_df.inventory_units

    # Define the hyperparameters to search over
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]

    model = ARIMA(train_df['inventory_units'], order=(1,1,1))
    result = model.fit()


    # we need to prepare the test data 
    # and fill empty values before we can submit it
    test_dataset = dataset_path + 'test.csv'
    test_df = pd.read_csv(test_dataset)
    test_df[['year_week', 'product_number']] = test_df.id.str.split('-', expand = True)
    test_df['product_number'] = test_df.product_number.astype(int)
    test_df

    # each product has specific information, we are just creating a reference table
    product_mapping = train_df[['product_number', 'prod_category', 'segment', 'specs', 'display_size']].drop_duplicates()
    product_mapping

    test_df_complete = test_df.merge(product_mapping, on='product_number', how = 'left')
    test_df_complete

    X_test = prepare_df(test_df_complete)
    y_pred = lm_model.predict(X_test)

    # this is an example submission
    submission = pd.DataFrame({
        'id' : test_df_complete.id,
        'inventory_units' : y_pred
    })
    
if __name__ == "__main__":
    main()