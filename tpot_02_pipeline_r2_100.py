import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator
from xgboost import XGBRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.10331629351564468
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=50, p=1, weights="uniform")),
        StackingEstimator(estimator=make_pipeline(
            Nystroem(gamma=0.15000000000000002, kernel="rbf", n_components=1),
            XGBRegressor(learning_rate=0.01, max_depth=7, min_child_weight=7, n_estimators=100, nthread=1, objective="reg:squarederror", subsample=0.4)
        ))
    ),
    OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10),
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=50, p=1, weights="uniform")),
    RandomForestRegressor(bootstrap=False, max_features=0.15000000000000002, min_samples_leaf=20, min_samples_split=15, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
