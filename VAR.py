# general libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os 
# RMSE 
from sklearn.metrics import mean_squared_error
# stationarity checks
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
# heatmap 
import seaborn as sns
# Vector Autoregression
from statsmodels.tsa.api import VAR

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

# We can also print the Auto Correlation Function (ACF): IT SHOWS THAT LAG 1 IS ALREADY MEANINGFUL
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

##### AT THIS POINT OUR DATA IS ALREADY STATIONARY! #####

# it's interesting to know the correlation among variables
corr_matrix = train_df.corr(numeric_only=True)
# plot it as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

# MODELING
train_diff = pd.get_dummies(train_diff, columns=['product_number', 'segment', 'prod_category']).astype(int)

model = VAR(train_diff)
results = model.fit(5)
results.summary()
