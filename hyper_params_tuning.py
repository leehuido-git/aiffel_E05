import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

random_state=2020

def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

def my_GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5):
    # GridSearchCV 모델로 초기화
    grid_model = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=verbose, n_jobs=n_jobs)
    # 모델 fitting
    grid_model.fit(train, y)
    # 결과값 저장
    params = grid_model.cv_results_['params']
    score = grid_model.cv_results_['mean_test_score']
    # 데이터 프레임 생성
    results = pd.DataFrame(params)
    results['score'] = score
    # RMSLE 값 계산 후 정렬
    results['RMSLE'] = np.sqrt(-1 * results['score'])
    results = results.sort_values('RMSLE')
    return results

def main():
    data_dir = os.path.join(local_path, 'data')

    train_data_path = os.path.join(data_dir, 'train.csv')
    test_data_path = os.path.join(data_dir, 'test.csv') 

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    train['date'] = train['date'].apply(lambda i: i[:6]).astype(int)
    test['date'] = test['date'].apply(lambda i: i[:6]).astype(int)

    ##############################################data preprocessing
    y_train = train['price']
    y_train = np.log1p(y_train)
    del train['price']
    del train['id']
    x_train = train

    skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_lot15', 'sqft_living15']
    for c in skew_columns:
        train[c] = np.log1p(train[c].values)
        test[c] = np.log1p(test[c].values)

    x_train, _, y_train, _ = train_test_split(x_train, y_train, random_state=random_state, test_size=0.2)

    ##############################################hyper parms tuning
    model = XGBRegressor(random_state=random_state)
    tuning_result = my_GridSearch(model, x_train, y_train, param_grid, verbose=2, n_jobs=-1)

    with open(os.path.join(data_dir, 'hyper_params.json'), 'w') as f_json:
        temp_result = dict()
        for key, val in tuning_result.iloc[0].to_dict().items():
            if type(val) == np.int64:
                temp_result[key] = int(val)
            elif type(val) == np.float64:
                temp_result[key] = float(val)
            else:
                temp_result[key] = val
        json.dump(temp_result, f_json)

if __name__ == '__main__':
    local_path = os.getcwd()
    param_grid = {
    'objective': ['reg:squarederror'],
    'base_score': [None],
    'booster': [None],
    'colsample_bylevel': [None],
    'colsample_bynode': [None],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'gamma': [0, 1, 2, 3, 4, 5],
    'gpu_id': [0],
    'importance_type': [None],
    'interaction_constraints': [None],
    'learning_rate': [0.4, 0.3, 0.2 ,0.1, 0.05, 0.03, 0.01],
    'max_delta_step': [None],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_child_weight': [None],
    'monotone_constraints': [None],
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800],
    'num_parallel_tree': [None],
    'predictor': [None],
    'random_state': [None],
    'reg_alpha': [None],
    'reg_lambda': [None],
    'scale_pos_weight': [None],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1],
    'tree_method': [None],
    'validate_parameters': [None],
    'verbosity': [None]
    }
    main()