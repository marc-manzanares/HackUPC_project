# just importing some general libraries
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os 

# it is useful to set the project directory to the data folder 
os.chdir('dataset/')

# DATA ANALYSIS
train_df = pd.read_csv('train.csv')

train_df

# quickly and without looking we'll just drop any NA values
train_df = train_df.dropna()

# we'll also remove duplicates based on arbitrary columns 
train_df = train_df.drop_duplicates(['id', 'year_week', 'product_number'])

# it is useful to set the correct type for the date for some plotting functions
train_df['date'] = pd.to_datetime(train_df['date'])

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

# it's interesting to know the correlation among variables
train_df.corr(numeric_only = True)

# MODELING
# our very simple ultra small data prep function
def prepare_df(df):
    df = df[['year_week', 'product_number', 'prod_category', 'segment']]
    df = pd.get_dummies(df, columns=['product_number', 'segment', 'prod_category'])
    return df

X_train = prepare_df(train_df)
y_train = train_df.inventory_units

y_train

# the best model in the world
lm_model = LinearRegression()
lm_model.fit(X_train, y_train)

y_pred = lm_model.predict(X_train)
y_true = y_train

# just checking the error on the training predictions
rms = mean_squared_error(y_true, y_pred, squared=False)
rms

# TEST DATA PREDICTIONS
# this is an example on how data must be submitted
pd.read_csv('sample_submission.csv')

# we need to prepare the test data 
# and fill empty values before we can submit it
test_df = pd.read_csv('test.csv')
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

submission

# to submit a kaggle notebook you must save the submission.csv file inside /kaggle/working directory 
# remember to skip the index before writing the file
submission.to_csv('dummy.csv', index = False)