import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

current_path = os.getcwd()
dataset_path=current_path + "/dataset/"

def clean_dataset(dataframe):

    # Load the dataset
    df = pd.read_csv(dataframe)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop missing values
    df.dropna(inplace=True)

    # Check for errors and inconsistencies
    # and perform necessary cleaning steps

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    df['normalized_sales_units'] = scaler.fit_transform(df[['sales_units']])
    df['normalized_inventory_units'] = scaler.fit_transform(df[['inventory_units']])

    df = df.drop(columns=['sales_units', 'inventory_units'])
    print(df)

    # Return the cleaned dataset
    return df


def main():
    cleaned_df = clean_dataset(dataset_path + 'train.csv')

if __name__ == "__main__":
    main()