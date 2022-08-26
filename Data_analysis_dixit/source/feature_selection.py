import pandas as pd
import config
import data_combine

def feature_select_quality(df):
    # combining and reading the data
    df.rename(columns = {' ': 'Output'}, inplace=True)
    df = pd.concat([df.loc[:,'300.377':'500.556'], df.loc[:,'Output']], axis=1)

    # removing calibratiion rows
    df_without_calb = df[~df['Output'].str.contains(('Calibration|CALIBARTION|CALIBARATION'),case=False, regex=True, na = False)]
    df_without_calb.reset_index(drop=True)

    # taking only F data
    df_with_F = df_without_calb[df_without_calb['Output'].str.split('_').str[0] == 'F']
    df_with_F.reset_index(drop=True)
    

    # creating column with quality of object
    df_with_F['quality'] = df_with_F['Output'].str.extract(pat=r'(Faint|None|Medium|Very Strong)', expand=False)
    df_with_F.drop(columns=['Output'], inplace=True)
    df_with_F.dropna(inplace=True)
    df_with_F.reset_index(drop=True, inplace=True)
    
    return df_with_F


def feature_select_color(df):
    # combining and reading the data
    df.rename(columns = {' ': 'Output'}, inplace=True)
    df = pd.concat([df.loc[:,'300.377':'500.556'], df.loc[:,'Output']], axis=1)
    # removing calibratiion rows
    df_without_calb = df[~df['Output'].str.contains(('Calibration|CALIBARTION|CALIBARATION'),case=False, regex=True, na = False)]
    df_without_calb.reset_index(drop=True)

    df_with_w = df_without_calb[df_without_calb['Output'].str.split('_').str[0] == 'W']
    df_with_w.reset_index(drop=True)

    # cleaning data
    df_with_w['quality'] = df_with_w['Output'].str.extract(pat=r'(Faint|None|Medium|Very Strong)', expand=False)
    df_with_w['color'] = df_with_w['Output'].str.extract(pat=r'(_[D-M]_|_[D-M]/|_[D-M] |_ [D-M])', expand=False, ).str.extract(r'([D-M])')
    df_with_w.dropna(inplace=True)
    df_with_w.reset_index(drop=True, inplace=True)

    df_with_w.drop(columns=['Output'], inplace=True)

    return df_with_w