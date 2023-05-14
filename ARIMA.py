# general libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os 
import sys
# RMSE 
from sklearn.metrics import mean_squared_error
# stationarity and ARMA variables checks
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# heatmap 
import seaborn as sns
# Model
import statsmodels.api as sm


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

# let's create some grouped tables
gd_product_number = train_df.groupby(['date', 'product_number']).sum(numeric_only = True).reset_index()
products = gd_product_number.product_number.unique()
gd_prod_category = train_df.groupby(['date', 'prod_category']).sum(numeric_only = True).reset_index()
prod_categories = gd_prod_category.prod_category.unique()

for prod in prod_categories: 
    gd_prod = gd_prod_category[gd_prod_category.prod_category == prod]
    plt.plot(gd_prod.date, gd_prod.inventory_units)
    
plt.show()

for prod in products: 
    gd_prod = gd_product_number[gd_product_number.product_number == prod]
    plt.plot(gd_prod.date, gd_prod.inventory_units)
    
plt.xticks(rotation = 45)    
plt.show()

gd_product_number = gd_product_number[gd_product_number.product_number == 233919]
for prod in products: 
    gd_prod = gd_product_number[gd_product_number.product_number == prod]
    plt.plot(gd_prod.date, gd_prod.inventory_units)
    
plt.xticks(rotation = 45)    
plt.show()

plt.hist(train_df.inventory_units, bins = 30)
plt.show()

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.dropna(inplace=True)

# MODELING
# our very simple ultra small data prep function
def prepare_df(df):
    df = df[['year_week', 'product_number', 'prod_category', 'segment']]
    df = pd.get_dummies(df, columns=['product_number', 'segment', 'prod_category'])
    return df

train_df_1 = prepare_df(train_df)
train_df_1.to_csv("Xtrain.csv")

# Select the endogenous variable
endog = train_df[['inventory_units']]
y_true = train_df[['id', 'inventory_units', 'product_number']]

# Create the exogenous variables (if any)
exog = train_df[['sales_units']]

# Set frequency to daily
endog.index.freq = 'W'
exog.index.freq = 'W'

# Create ARMA model
#model = sm.tsa.SARIMAX(endog=endog, exog=exog, order=(1, 0, 4), seasonal_order=(1, 0, 4, 52))
model = sm.tsa.ARIMA(endog=endog, exog=exog, order=(1, 0, 4))

# Fit the model
results = model.fit()

# print the summary of the model
print(results.summary())

test_df = pd.read_csv('test.csv')
test_df[['year_week', 'product_number']] = test_df.id.str.split('-', expand = True)
test_df['product_number'] = test_df.product_number.astype(int)

# each product has specific information, we are just creating a reference table
product_mapping = train_df[['product_number', 'prod_category', 'segment', 'specs', 'display_size']].drop_duplicates()

test_df_complete = test_df.merge(product_mapping, on='product_number', how = 'left')

X_test = prepare_df(test_df_complete)
print(test_df_complete)

# reshape X_test to match expected shape

# make predictions
print(y_true)
print(test_df)

y_pred = results.predict(start=0, end=len(X_test)-1, exog=X_test)

merged_df = pd.merge(y_true, test_df, on ='product_number')
print(merged_df['inventory_units'])
# Take the real y_true values (train inventory_units)
# merge the two dataframes on the 'id' column
#merged_df = pd.merge(y_true, test_df, on='id')
#asdasdresult = merged_df['inventory_units']
#print(merged_df['inventory_units'].values)

#rmse = np.sqrt(mean_squared_error(y_true, y_pred, squared=False))

#print(rmse)

# this is an example submission
submission = pd.DataFrame({
    'id' : test_df_complete.id,
    'inventory_units' : y_pred
})

#submission.dropna(subset=['id', 'inventory_units'], how='any', inplace=True)

# to submit a kaggle notebook you must save the submission.csv file inside /kaggle/working directory 
# remember to skip the index before writing the file
submission.to_csv('submission.csv', index = False)

#predictions = model_fit.predict(start=test.index[0], end=test.index[-1], exog=test[['exog1', 'exog2']])
