import argparse
from ipaddress import collapse_addresses
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

    # if you see the column name for your output then please replace this 
    df.rename(columns = {' ': 'Output'}, inplace=True)
    # df.drop(columns=['Unnamed: 1059'], inplace=True)

    # Removing Calibration rows
    df_without_calb = df[~df['Output'].str.contains(('Calibration|CALIBARTION|CALIBARATION'),case=False, regex=True, na = False)]
    df_without_calb.reset_index(drop=True, inplace=True)

    # Select Range from 300 to 500 for color pred
    df_test = pd.concat([df_without_calb.loc[:,'300.377':'500.556'],df_without_calb['Output']],axis=1)

    # Select Range from 410 to 570 for color pred
    df_test_quality = pd.concat([df_without_calb.loc[:,'409.561':'570.067'],df_without_calb['Output']],axis=1)

    # Taking Only W obsercations
    df_with_w = df_test[df_test['Output'].str.split('_').str[0] == 'W']
    df_with_f = df_test_quality[df_test_quality['Output'].str.split('_').str[0] == 'F']

    return df_with_w,df_with_f


def predict(df_pred, df_quality):
    x_qual = df_quality.drop(columns = ['Output'])
    x_qual.reset_index(drop = True, inplace = True)
    qual_mod = joblib.load(f'{config.saved_model}/quality_extra_trees_97.77.pkl')
    quality_pred = pd.DataFrame(qual_mod.predict(x_qual))

    # Dropping Output Value and color column
    x = df_pred.drop(columns=['Output'])
    x.reset_index(drop=True, inplace=True)

    enc = joblib.load(f'{config.saved_model}/enc.joblib')

    x_trans = pd.DataFrame(enc.transform(quality_pred).toarray())
    X = pd.concat([x, x_trans], axis=1, ignore_index=True)

    # Add your file path here in load method
    extra_tree = joblib.load(f'{config.saved_model}/f1_85.18_4_08.pkl')

    y_preds = extra_tree.predict(X)
    y_prob = extra_tree.predict_proba(X)

    print('Probilities with respect to classes', '\n', y_prob[-1],'\n')
    print('Quality of Object is "',quality_pred[0].iloc[-1],'" \n', 'Color of the Object is "',y_preds[-1],'"\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--test",
    type=str,
    default='29-06-2022-02-45-53'
    )

    args = parser.parse_args()

    df_pred, df_quality= Preprocessing(
        test = args.test
    )
    predict(
        df_pred=df_pred, df_quality=df_quality
    )