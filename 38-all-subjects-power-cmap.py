import matplotlib
import glob
import pandas as pd
import mne
import matplotlib.pyplot as plt
from io import StringIO
import mne
from mne.io import read_raw_eeglab, read_epochs_eeglab
import numpy as np
from scipy import signal
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm_notebook
from sklearn.svm import SVR
import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import multiprocessing
from tpot import TPOTRegressor
from oct2py import octave
from joblib import Parallel, delayed
import multiprocessing
from joblib import wrap_non_picklable_objects
import json
import pickle
import os.path
from mpl_toolkits.mplot3d import axes3d
from mayavi import mlab

print(matplotlib.get_backend())

# matplotlib.use('Qt5Agg')
matplotlib.interactive(True)

features_filename = '36-features.xlsx'
df = pd.read_excel(features_filename, index_col=0)

df2 = df
bins = [0, 4, 8, 12, 20, 30, 46, 70]

def categorize_by_freq_time(df2, bins = [0, 4, 8, 12, 20, 30, 46, 70], time_bins=np.linspace(-500, 0, 11)):
    df2['wavelet_all_max_power_at_freq_bin'] = pd.cut(df2['wavelet_all_max_power_at_freq'], bins)
    df2['wavelet_all_max_power_at_time_bin'] = pd.cut(df2['wavelet_all_max_power_at_time'], time_bins/1000)
    df2['wavelet_ltm1_max_power_at_freq_bin'] = pd.cut(df2['wavelet_ltm1_max_power_at_freq'], bins)
    df2['wavelet_ltm1_max_power_at_time_bin'] = pd.cut(df2['wavelet_ltm1_max_power_at_time'], time_bins/1000)
    df2['wavelet_rtm1_max_power_at_freq_bin'] = pd.cut(df2['wavelet_rtm1_max_power_at_freq'], bins)
    df2['wavelet_rtm1_max_power_at_time_bin'] = pd.cut(df2['wavelet_rtm1_max_power_at_time'], time_bins/1000)
    return df2

def groupby_freq_time(df):
    df = pd.DataFrame(df.to_records())
    df['wavelet_ltm1_max_power_at_freq_bin_left'] = df['wavelet_ltm1_max_power_at_freq_bin'].apply(lambda x: x.left)
    df['wavelet_ltm1_max_power_at_time_bin_left'] = df['wavelet_ltm1_max_power_at_time_bin'].apply(lambda x: x.left)
    df = df.fillna(0)
    return df


df_percentile_small = categorize_by_freq_time(df2[df2['mep_category_percentile'] == 0]).groupby(['wavelet_ltm1_max_power_at_freq_bin', 'wavelet_ltm1_max_power_at_time_bin']).count()
df_percentile_small = groupby_freq_time(df_percentile_small)

df_percentile_medium = categorize_by_freq_time(df2[df2['mep_category_percentile'] == 1]).groupby(['wavelet_ltm1_max_power_at_freq_bin', 'wavelet_ltm1_max_power_at_time_bin']).count()
df_percentile_medium = groupby_freq_time(df_percentile_medium)

df_percentile_large = categorize_by_freq_time(df2[df2['mep_category_percentile'] == 2]).groupby(['wavelet_ltm1_max_power_at_freq_bin', 'wavelet_ltm1_max_power_at_time_bin']).count()
df_percentile_large = groupby_freq_time(df_percentile_large)


time_width = 0.02
freq_width = 1.2
fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection='3d')
ax.bar3d(df_percentile_small['wavelet_ltm1_max_power_at_freq_bin_left'], df_percentile_small['wavelet_ltm1_max_power_at_time_bin_left'], np.zeros_like(df_percentile_small['all_all_max_amplitude']), freq_width, time_width, df_percentile_small['all_all_max_amplitude'], shade=True)
ax.bar3d(df_percentile_medium['wavelet_ltm1_max_power_at_freq_bin_left'] + freq_width, df_percentile_medium['wavelet_ltm1_max_power_at_time_bin_left'] + time_width, np.zeros_like(df_percentile_medium['all_all_max_amplitude']), freq_width, time_width, df_percentile_medium['all_all_max_amplitude'], shade=True)
ax.bar3d(df_percentile_large['wavelet_ltm1_max_power_at_freq_bin_left'] + freq_width * 2, df_percentile_large['wavelet_ltm1_max_power_at_time_bin_left'] + time_width * 2, np.zeros_like(df_percentile_large['all_all_max_amplitude']), freq_width, time_width, df_percentile_large['all_all_max_amplitude'], shade=True)
ax.set_xlabel('Freq - Ltm1')
ax.set_ylabel('Time - Ltm1')
ax.set_zlabel('MEP category (percentile)')
plt.show(block=True)

# df_percentile_small = categorize_by_freq_time(df2[df2['mep_category_percentile'] == 0]).groupby(['wavelet_ltm1_max_power_at_freq_bin', 'wavelet_ltm1_max_power_at_time_bin']).count()
# df_percentile_small = groupby_freq_time(df_percentile_small)
# fig = plt.figure(figsize=(9,6))
# ax = plt.axes(projection='3d')
# ax.bar3d(df_percentile_small['wavelet_ltm1_max_power_at_freq_bin_left'], df_percentile_small['wavelet_ltm1_max_power_at_time_bin_left'], np.zeros_like(df_percentile_small['all_all_max_amplitude']), 0.5, 0.005, df_percentile_small['all_all_max_amplitude'], shade=True)
# ax.set_xlabel('Freq - Ltm1')
# ax.set_ylabel('Time - Ltm1')
# ax.set_zlabel('MEP category (percentile)')