import argparse
import os

import re
import config

import numpy as np
import pandas as pd

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import warnings
import joblib
warnings.filterwarnings("ignore")

def Preprocessing(test):
    df = pd.read_csv(f'{config.test_data}/{test}.csv')

    # if you see the column name for your output then 
    df.rename(columns = {'Unnamed: 1': 'Output'}, inplace=True)
    df.drop(columns=['Unnamed: 0','Unnamed: 1059'], inplace=True)

    # Select Range from 300 to 500
    df_test = pd.concat([df.loc[:,'300.377':'500.556'], df.loc[:,'Output']], axis=1)

    # Removing Calibration rows
    df_without_calb = df_test[~df_test['Output'].str.contains(('Calibration|CALIBARTION|CALIBARATION'),case=False, regex=True, na = False)]
    df_without_calb.reset_index(drop=True)

    # Taking Only W obsercations
    df_with_w = df_without_calb[df_without_calb['Output'].str.split('_').str[0] == 'W']

    # Creating Color and quality columns
    df_with_w['quality'] = df_with_w['Output'].str.extract(pat=r'(Faint|None|Medium|Very Strong)', expand=False)
    df_with_w['color'] = df_with_w['Output'].str.extract(pat=r'(_[D-M]_|_[D-M]/|_[D-M] |_ [D-M])', expand=False, ).str.extract(r'([D-M])')

    return df_with_w


def predict(df_pred):

    y = df_pred['color']

    # Dropping Output Value and color column
    x = df_pred.drop(columns=['Output', 'color'])
    df_pred.reset_index(drop=True, inplace=True)

    enc = joblib.load(f'{config.saved_model}/enc.joblib')

    x_trans = pd.DataFrame(enc.transform(x[['quality']]).toarray())
    X = pd.concat([x.iloc[:,:-1], x_trans], axis=1)

    # Add your file path here in load method
    extra_tree = joblib.load(f'{config.saved_model}/f1_85.18_4_08.pkl')

    y_preds = extra_tree.predict(X)
    
    print(y_preds[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--test",
    type=str,
    default='test_data_1'
    )

    args = parser.parse_args()

    df_pred = Preprocessing(
        test = args.test
    )
    predict(
        df_pred
    )