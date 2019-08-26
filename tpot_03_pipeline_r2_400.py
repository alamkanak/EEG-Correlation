import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, StandardScaler
from tpot.builtins import OneHotEncoder, StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.08311510657219474
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.01, loss="quantile", max_depth=10, max_features=0.05, min_samples_leaf=5, min_samples_split=2, n_estimators=100, subsample=0.7000000000000001)),
    OneHotEncoder(minimum_fraction=0.15, sparse=False, threshold=10),
    Normalizer(norm="l2"),
    OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10),
    Normalizer(norm="l2"),
    StandardScaler(),
    RandomForestRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=15, min_samples_split=2, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
