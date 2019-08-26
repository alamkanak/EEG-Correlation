import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-0.7866397807079906
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=46, p=1, weights="uniform")),
    RobustScaler(),
    StandardScaler(),
    StackingEstimator(estimator=LinearSVR(C=15.0, dual=True, epsilon=0.01, loss="squared_epsilon_insensitive", tol=0.001)),
    MaxAbsScaler(),
    RandomForestRegressor(bootstrap=False, max_features=0.15000000000000002, min_samples_leaf=18, min_samples_split=8, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
