import argparse
import os

import numpy as np
import pandas as pd

import config
import model_dispatcher
import feature_selection
from data_combine import data_gathering

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import warnings
import joblib
warnings.filterwarnings("ignore", category=FutureWarning)


def run(fold, model, type):
    warnings.filterwarnings("ignore")
    skf = StratifiedKFold(n_splits=fold, shuffle=True,random_state=1)
    
    df = data_gathering(config.data_path)
    
    if type == 'color':
        data_train = feature_selection.feature_select_color(df)
        enc_ = joblib.load(config.encoder_data)
        x_trans = pd.DataFrame(enc_.transform(data_train[['quality']]).toarray())
        X = pd.concat([data_train.iloc[:,:-2], x_trans], axis=1)
        Y = data_train.iloc[:,-1]
    
    elif type == 'quality':
        data_train = feature_selection.feature_select_quality(df)
        X = data_train.iloc[:,:-1]
        Y = data_train.iloc[:,-1]

   
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state =42, stratify = Y)

    lst_accu_stratified = []
    lst_accu_stratified_train = []

    for train_index, test_index in skf.split(X_train, Y_train):
        x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = Y.iloc[train_index], Y.iloc[test_index]

        clf = model_dispatcher.models[model]
        clf.fit(x_train_fold, y_train_fold)
        Y_pred = clf.predict(x_test_fold)

        lst_accu_stratified.append(metrics.f1_score(y_test_fold,Y_pred,average='weighted'))

        lst_accu_stratified_train.append(clf.score(x_train_fold, y_train_fold))
    print(f"train_accuracy={lst_accu_stratified_train}")
    print(f"f1_score={lst_accu_stratified}")
    print(f"f1 socre train_accuracy={np.mean(lst_accu_stratified_train)*100}")
    print(f"f1_score_mean={np.mean(lst_accu_stratified)*100}")

    # joblib.dump(
    #     clf,  
    #     os.path.join(config.saved_model, f"f1_{round(np.mean(lst_accu_stratified)*100,2)}_{fold}_folds.pkl") 
    # )


    y_test_pred = clf.predict(X_test)
    print(f"f1_score_test_data: {metrics.f1_score(Y_test,y_test_pred,average='weighted')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--fold",
    type=int,
    default=5
    )

    parser.add_argument(
    "--type",
    type=str,
    default='color'
    )

    parser.add_argument(
    "--model",
    type=str,
    default='et'
    )
    args = parser.parse_args()
    run(
    fold=args.fold,
    model=args.model,
    type=args.type
    )