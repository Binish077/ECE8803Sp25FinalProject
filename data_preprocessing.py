# PROVIDED IN STARTING KIT, NOT IN USE
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from PIL import Image

def combine_excel(csv_dir):
    filenames = glob.glob(csv_dir + "/*.xlsx")
    outputxlsx = pd.DataFrame()

    for file in filenames:
        df = pd.concat(pd.read_excel(file, sheet_name=None), ignore_index=True, sort=False)
        outputxlsx = outputxlsx.append(df, ignore_index=True)

    outputxlsx.to_csv('test_set_labels.csv',index=False)

def analyze_dataframe(csv_dir):
    pass

def process_images(csv_dir):
    df = pd.read_csv(csv_dir)

    for i in tqdm(range(0,len(df))):
        path = df.iloc[i,0]
        im = Image.open(path).convert('L')



if __name__ == '__main__':
    csv_dir = '/home/kiran/Desktop/Dev/VIPCUP2023_OLIVES/csv_dir/Training_Biomarker_Data.csv'
    df = pd.read_csv(csv_dir)
    print(df[df['B1'].isna()])
    print(df[df['B2'].isna()])
    print(df[df['B3'].isna()])
    print(df[df['B4'].isna()])
    print(df[df['B5'].isna()])
    print(df[df['B6'].isna()])


    #process_images(csv_dir)