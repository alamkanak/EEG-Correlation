import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-0.4648468109523711
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            FunctionTransformer(copy),
            FunctionTransformer(copy)
        ),
        FunctionTransformer(copy)
    ),
    SelectPercentile(score_func=f_regression, percentile=6),
    FeatureAgglomeration(affinity="cosine", linkage="complete"),
    MinMaxScaler(),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.001, loss="quantile", max_depth=1, max_features=0.15000000000000002, min_samples_leaf=9, min_samples_split=12, n_estimators=100, subsample=1.0)),
    RobustScaler(),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.05, min_samples_leaf=2, min_samples_split=8, n_estimators=100)),
    RandomForestRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=20, min_samples_split=8, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
