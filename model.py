# general libraries
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import os 
# RMSE 
from sklearn.metrics import mean_squared_error
# stationarity and ARMA variables checks
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# it is useful to set the project directory to the data folder 
os.chdir('dataset/')

# https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide
# product_number == family
# reporterhq_id == store_nbr

# Import train dataset
train_df =  pd.read_csv("train.csv")

# drop any NA values 
train_df = train_df.dropna()

# set correct type for the date
train_df["date"] = pd.to_datetime(train_df.date)

# Examining sales with stores: 
# https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide?scriptVersionId=78418304&cellId=26
a = train_df[["reporterhq_id", "sales_units"]]
a["ind"] = 1
a["ind"] = a.groupby("reporterhq_id").ind.cumsum().values
a = pd.pivot(a, index = "ind", columns = "reporterhq_id", values = "sales_units").corr()
mask = np.triu(a.corr())
plt.figure(figsize=(20, 20))
sns.heatmap(a,
        annot=True,
        fmt='.1f',
        cmap='coolwarm',
        square=True,
        mask=mask,
        linewidths=1,
        cbar=False)
plt.title("Correlations among stores",fontsize = 20)
plt.show()

# Let's check weekly total sells
a = train_df.set_index("date").groupby("reporterhq_id").resample("W").sales_units.sum().reset_index()
fig = px.line(a, x = "date", y= "sales_units", color = "reporterhq_id", title = "Weekly total sales of each store")
fig.show()

# Higher sales in stores 3, 24, and 15

# There are some stores that where not opened at the beginning, we can remove those rows
# https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide?scriptVersionId=78418304&cellId=31
# TODO: comprovar quan va obrir cada tenda i eliminar files d'abans d'apertura si existeixen

# Some product families depends on seasonality
# TODO: no sé perquè printa quan sales != 0
c = train_df.groupby(["prod_category", "reporterhq_id"]).tail(60).groupby(["prod_category", "reporterhq_id"]).sales_units.sum().reset_index()
c[c.sales_units == 0.0]
print(c)

# weekly total sales of the product category
a = train_df.set_index("date").groupby("prod_category").resample("W").sales_units.sum().reset_index()
fig = px.line(a, x = "date", y= "sales_units", color = "prod_category", title = "Weekly total sales of each product category")
fig.show()

# Most sold categories: Clover, Goku and Doraemon
# We can also make a barplot
a = train_df.groupby("prod_category").sales_units.mean().sort_values(ascending = True).reset_index()
fig = px.bar(a, y = "prod_category", x="sales_units", color = "prod_category", title = "Which family product is preferred more?")
fig.show()
# TODO: aquí surten altres, wtf

# How different can stores be from each other?
d = train_df
d["reporterhq_id"] = d["reporterhq_id"].astype("int8")
d["year"] = d.date.dt.year
fig = px.line(d.groupby(["reporterhq_id", "year"]).sales_units.mean().reset_index(), x = "year", y = "sales_units", color = "reporterhq_id")
fig.show()
# TODO: aquí es veu bé quan va començar cada tenda per eliminar files

# me he quedado aquí: https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide?scriptVersionId=78418304&cellId=51