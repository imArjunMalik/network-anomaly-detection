"""
This file is used to create the training dataset for the model. This is done 
because the dataset is highly imbalanced. 

"""


import pandas as pd
from sklearn.utils import resample
import os

def load_data(input_file):
    df = pd.read_csv(input_file)
    return df

def balance_data(df):
    # Separate majority and minority classes
    df_majority = df[df.Label == 1]  # Assuming 'BENIGN' is encoded as 1
    df_minority = df[df.Label == 0]  # Assuming 'ANOMALOUS' is encoded as 0

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,    # sample without replacement
                                       n_samples=len(df_minority),  # to match minority class
                                       random_state=42)  # reproducible results

    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    return df_balanced

def save_data(df, output_file):
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_path, '../data/processed/processed_data.csv')
    output_file = os.path.join(base_path, '../data/processed/train.csv')

    df = load_data(input_file)
    df_balanced = balance_data(df)
    save_data(df_balanced, output_file)
