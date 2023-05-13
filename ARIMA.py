# general libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os 
# RMSE 
from sklearn.metrics import mean_squared_error
# stationarity and ARMA variables checks
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# heatmap 
import seaborn as sns
# Model
from statsmodels.tsa.statespace.sarimax import SARIMAX

# it is useful to set the project directory to the data folder 
os.chdir('dataset/')

# DATA ANALYSIS
train_df = pd.read_csv('train.csv')
print(train_df.head())

# quickly and without looking we'll just drop any NA values
train_df = train_df.dropna()

# we'll also remove duplicates based on arbitrary columns 
train_df = train_df.drop_duplicates(['id', 'year_week', 'product_number'])

# it is useful to set the correct type for the date for some plotting functions
train_df['date'] = pd.to_datetime(train_df['date'])

# set id variable to integer
train_df['id'] = train_df['id'].str.replace('-', '').astype(int)
print(train_df.head())

gd = train_df.groupby('date').sum(numeric_only = True)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(gd.index, gd.inventory_units)
ax1.set(xlabel='Date', ylabel='Inventory Units')
ax1.set_title('Inventory in time')

ax2.plot(gd.index, gd.sales_units)
ax1.set(xlabel='Date', ylabel='Sales Units')
ax2.set_title('Sales in time')
plt.show()

# Perfom KPSS test to check stationarity, we want p-value to be > 0.05
inv_res = kpss(train_df['inventory_units'])
print('KPSS Statistic: {:.3f}'.format(inv_res[0]))
print('p-value: {:.3f}'.format(inv_res[1]))
print('Lags Used: {}'.format(inv_res[2]))
print('Critical Values:')
for key, value in inv_res[3].items():
    print('\t{}: {:.3f}'.format(key, value))

inv_res = kpss(train_df['sales_units'])
print('KPSS Statistic: {:.3f}'.format(inv_res[0]))
print('p-value: {:.3f}'.format(inv_res[1]))
print('Lags Used: {}'.format(inv_res[2]))
print('Critical Values:')
for key, value in inv_res[3].items():
    print('\t{}: {:.3f}'.format(key, value))

# We can also print the Partial Auto Correlation Function (PACF) and know the lag for MA
# It shows that we should pick p = 1
pacf_plot = plot_pacf(train_df['inventory_units'])
x_limits = pacf_plot.axes[0].get_xlim()
# Set the x-axis limits to a new range of values
pacf_plot.axes[0].set_xlim([0, 4])
plt.show()

# We can also print the Auto Correlation Function (ACF) and know the lag for AR
# It shows that we should pick q = 4
acf_plot = plot_acf(train_df['inventory_units'])
x_limits = acf_plot.axes[0].get_xlim()
# Set the x-axis limits to a new range of values
acf_plot.axes[0].set_xlim([0, 4])
plt.show()

# We apply differencing of lag 1
train_diff = train_df
train_diff['inventory_units'] = train_df['inventory_units'].diff()
train_diff['sales_units'] = train_df['sales_units'].diff()
# drop NA values again
train_diff = train_diff.dropna()

# and check stationarity again
inv_res = kpss(train_diff['inventory_units'])
print('KPSS Statistic: {:.3f}'.format(inv_res[0]))
print('p-value: {:.3f}'.format(inv_res[1]))
print('Lags Used: {}'.format(inv_res[2]))
print('Critical Values:')
for key, value in inv_res[3].items():
    print('\t{}: {:.3f}'.format(key, value))

inv_res = kpss(train_diff['sales_units'])
print('KPSS Statistic: {:.3f}'.format(inv_res[0]))
print('p-value: {:.3f}'.format(inv_res[1]))
print('Lags Used: {}'.format(inv_res[2]))
print('Critical Values:')
for key, value in inv_res[3].items():
    print('\t{}: {:.3f}'.format(key, value))

# plot also to check
gd = train_diff.groupby('date').sum(numeric_only = True)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(gd.index, gd.inventory_units)
ax1.set(xlabel='Date', ylabel='Inventory Units')
ax1.set_title('Inventory in time')

ax2.plot(gd.index, gd.sales_units)
ax1.set(xlabel='Date', ylabel='Sales Units')
ax2.set_title('Sales in time')
plt.show()

# IT SHOWS WITH D = d = 1 IT IS ALREADY STATIONARY

# it's interesting to know the correlation among variables
corr_matrix = train_df.corr(numeric_only=True)
# plot it as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

# MODELING
def prepare_df(df):
    df = df[['year_week','reporterhq_id', 'prod_category', 'specs', 'segment']]
    df = pd.get_dummies(df, columns=['segment', 'prod_category']).astype(int)
    return df


train_df = prepare_df(train_df)
train_df.to_csv("Xtrain.csv")

# Create ARMA model
model = SARIMAX(train_df, order=(1, 0, 4), seasonal_order=(1, 0, 4, 52))

# Fit the model
results = model.fit()

# print the summary of the model
print(results.summary())
