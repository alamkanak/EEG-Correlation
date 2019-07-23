import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-0.49919708799243334
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.9500000000000001, min_samples_leaf=5, min_samples_split=7, n_estimators=100)),
            SelectPercentile(score_func=f_regression, percentile=35),
            OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10)
        ),
        FunctionTransformer(copy)
    ),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=2, min_child_weight=9, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.05)),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=2, p=1, weights="distance")),
    SelectFwe(score_func=f_regression, alpha=0.006),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=36, p=1, weights="uniform")),
    RandomForestRegressor(bootstrap=True, max_features=0.15000000000000002, min_samples_leaf=17, min_samples_split=2, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
