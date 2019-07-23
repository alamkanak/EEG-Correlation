import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator, ZeroCount

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.36603794750689644
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=4),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.85, learning_rate=0.01, loss="lad", max_depth=2, max_features=0.7000000000000001, min_samples_leaf=1, min_samples_split=2, n_estimators=100, subsample=0.05)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.25, min_samples_leaf=19, min_samples_split=8, n_estimators=100)),
    ZeroCount(),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=40, p=1, weights="uniform")),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=40, p=1, weights="uniform")),
    MinMaxScaler(),
    AdaBoostRegressor(learning_rate=0.001, loss="square", n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
