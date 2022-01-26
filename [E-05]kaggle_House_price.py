import os
import json
from matplotlib.font_manager import json_dump
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

random_state=2020

def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

def main():
    ##############################################data load
    data_dir = os.path.join(local_path, 'data')

    train_data_path = os.path.join(data_dir, 'train.csv')
    test_data_path = os.path.join(data_dir, 'test.csv') 

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    if os.path.isfile(os.path.join(data_dir, 'hyper_params.json')):
        with open(os.path.join(data_dir, 'hyper_params.json'), "r") as f_json:
            json_parms = json.load(f_json)
    else:
        json_parms = False
    ##############################################data preprocessing
    train['date'] = train['date'].apply(lambda i: i[:6]).astype(int)
    test['date'] = test['date'].apply(lambda i: i[:6]).astype(int)

    y_train = train['price']
    y_train = np.log1p(y_train)
    del train['price']
    del train['id']
    x_train = train

    skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_lot15', 'sqft_living15']
    for c in skew_columns:
        train[c] = np.log1p(train[c].values)
        test[c] = np.log1p(test[c].values)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=random_state, test_size=0.2)

    ###############################################parms load, test
    if json_parms is not False:
        del json_parms['score']
        del json_parms['RMSLE']
        print(json_parms)
        xgbr = XGBRegressor(json_parms, random_state=random_state)

    else:
        xgbr = XGBRegressor(random_state=random_state)

    xgbr.fit(x_train, y_train)
    y_pred = xgbr.predict(x_val)
    print(rmse(y_val, y_pred))

    prediction = xgbr.predict(test)
    prediction = np.expm1(prediction)

    submission_path = os.path.join(data_dir, 'sample_submission.csv')
    submission = pd.read_csv(submission_path)
    submission['price'] = prediction
    submission.to_csv(os.path.join(data_dir, 'sample_submission_{}.csv'.format(int(rmse(y_val, y_pred)))), index=False)

if __name__ == '__main__':
    local_path = os.getcwd()
    main()