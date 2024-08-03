"""
This file loads the raw files and combines them into one csv file. 

"""

# import libraries
import pandas as pd 
import os

def load_and_combine_csv(files, output_file):
    dataframes = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    return combined_df

if __name__ == "__main__":

    # csv_files = ['/Users/arjunmalik/Documents/College/UW Madison/Courses/Summer 2024/ECE 539/Final Project/anomaly-detection/data/raw/file-1.csv', 'data/raw/file-2.csv', 'data/raw/file-3.csv',
    #     'data/raw/file-4.csv', 'data/raw/file-5.csv', 'data/raw/file-6.csv', 
    #     'data/raw/file-7.csv', 'data/raw/file-8.csv']
    # combined_file = 'data/combined.csv'
    # load_and_combine_csv(csv_files, combined_file)
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_files = [
        os.path.join(base_path, '../data/raw/file-1.csv'),
        os.path.join(base_path, '../data/raw/file-2.csv'),
        os.path.join(base_path, '../data/raw/file-3.csv'),
        os.path.join(base_path, '../data/raw/file-4.csv'),
        os.path.join(base_path, '../data/raw/file-5.csv'),
        os.path.join(base_path, '../data/raw/file-6.csv'),
        os.path.join(base_path, '../data/raw/file-7.csv'),
        os.path.join(base_path, '../data/raw/file-8.csv'),
    ]
    combined_file = os.path.join(base_path, '../data/combined.csv')
    load_and_combine_csv(csv_files, combined_file)
