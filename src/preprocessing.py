"""
This file includes data preprocessing steps such as handling missing values, 
encoding the labels, and performing feature selection. 

"""
# import libraries 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # strip leading and trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    # remove missing or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Label anything not BENIGN as ANOMALOUS
    df['Label'] = df['Label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'ANOMALOUS')

    # Perform normalization
    X = df.iloc[:, :-1]  # all columns except the last one (features)
    y = df.iloc[:, -1]   # the last column (label)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # combine the scaled features and the label back into a single DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['Label'] = y.values

    # perform label encoding
    label_encoder = LabelEncoder()
    df_scaled['Label'] = label_encoder.fit_transform(df_scaled['Label'])
    
    # save the mapping of labels to numerical values
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    # save the processed data
    df_scaled.to_csv(output_file, index=False)

    return df_scaled, label_mapping

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_path, '../data/combined.csv')
    output_file = os.path.join(base_path, '../data/processed/processed_data.csv')
    df, label_mapping = preprocess_data(input_file, output_file)

    # print the label mapping
    print(label_mapping)


"""
{'ANOMALOUS': 0, 'BENIGN': 1}
"""
