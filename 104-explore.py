#%%
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
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from keras_tqdm import TQDMNotebookCallback
from tensorboard.plugins.hparams import api as hp
from livelossplot.tf_keras import PlotLossesCallback

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
import timeit
from skimage.transform import resize
from timeit import default_timer as timer
from datetime import timedelta
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.models import Model, load_model
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.utils import plot_model

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# import keras
# from keras import layers
# from keras.models import Model
# from keras import optimizers
# from keras import callbacks
# from keras.utils import plot_model

#%%
eegs = [
    'data/original/sub03/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub03/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub03/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
#     'data/original/sub03/exp02/eeg/SP 110RMT r1/clean-prestimulus.set', NO CMAP
#     'data/original/sub03/exp02/eeg/SP 110RMT r2/clean-prestimulus.set', NO CMAP
#     'data/original/sub03/exp03/eeg/SP 110RMT r1/clean-prestimulus.set', NO CMAP
    'data/original/sub03/exp03/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub03/exp03/eeg/SP 110RMT r3/clean-prestimulus.set',
#     'data/original/sub04/exp01/eeg/SP 110RMT r1/clean-prestimulus.set', NO CMAP
#     'data/original/sub04/exp01/eeg/SP 110RMT r2/clean-prestimulus.set', NO CMAP
#     'data/original/sub04/exp01/eeg/SP 110RMT r3/clean-prestimulus.set', NO CMAP
#     'data/original/sub05/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub06/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub06/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub06/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub06/exp02/eeg/SP 110RMT/clean-prestimulus.set',
    'data/original/sub07/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub07/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub07/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub08/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub08/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub08/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub08/exp02/eeg/SP 110RMT/clean-prestimulus.set',
    'data/original/sub08/exp03/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub08/exp03/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub08/exp03/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub12/exp02/eeg/SP 110RMT/clean-prestimulus.set',
#     'data/original/sub13/exp01/eeg/SP 110RMT/clean-prestimulus.set', LEFT HANDED
    'data/original/sub14/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub15/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub15/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub15/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub16/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub16/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub16/exp01/eeg/SP 110RMT r3/clean-prestimulus.set'
]

#%% [markdown]
# # Read and process EEG

#%%
def read_eeg(path):
    eeg = octave.pop_loadset(path)
    new_trial_list = []
    for i in range(eeg.data.shape[2]):
        trial = eeg.data[:, :, i]
        time = np.linspace(-1000, -20, num=trial.shape[1])
        trial = pd.DataFrame(np.transpose(trial), columns=eeg.chanlocs.labels[0])
        trial['time'] = time
        new_trial_list.append(trial)
    return new_trial_list

def crop_trials(trial_list, duration_millis=500, sampling_rate=2048):
    new_trial_list = []
    for trial in trial_list:
        samples_to_pick = duration_millis * sampling_rate / 1000
        new_trial_list.append(trial.tail(int(samples_to_pick)))
    return new_trial_list, samples_to_pick

def band_max(freq, psd, interval):
    indices = []
    for el in freq:
        indices.append(el in interval)
    freq = freq[indices]
    psd = psd[indices]
    if (len(psd) == 0):
        return 0, 0
    i = np.argmax(np.abs(psd))
    return freq[i], psd[i]

def read_wavelets(path):
    with open(path, 'rb') as f:
        wt = pickle.load(f)
    return wt

def wavelet_band_max(df, interval):
    indices = []
    for el in (df.index * 1000):
        indices.append(el in interval)
    df = df[indices]
    if (df.shape[0] == 0):
        return 0, 0, 0, 0
    return df.mean(axis=1).max(), df.mean(axis=1).argmax() * 1000, df.mean(axis=0).max(), df.mean(axis=0).argmax()

def crop_mep_region(mep_frame):
    crop_start = 0.211
    crop_end = 0.4
    multiplier = 7499 / 1.4998
    start = int(crop_start * multiplier)
    end = int(crop_end * multiplier)
    mep_cropped = mep_frame.iloc[start:end, :]
    time = mep_frame.iloc[start:end, :]
    return mep_cropped, time

def get_apb_column_name(mep_frame):
    if 'L APB' in mep_frame:
        return 'L APB'
    else:
        return 'APB'

def open_mep_as_df(path):
    fileMep = open(path, "r+")
    mep_frames = fileMep.read().split('\n\n')
    df_list = []
    for mep_frame in mep_frames:
        df_list.append(pd.read_csv(StringIO(mep_frame), '\t'))
    return df_list

def get_mep_path(eeg_path):
    segments = eeg_path.split('/')
    segments[4] = 'mep'
    segments = segments[:-1]
    mep_path = '/'.join(segments) + '/*.txt'
    mep_path = glob.glob(mep_path)[0]
    return mep_path

def calculate_mep_size(mep_frame):
    mep_cropped, time = crop_mep_region(mep_frame)
    apb_name = get_apb_column_name(mep_frame)
    max_row = mep_frame.iloc[mep_cropped.idxmax(axis=0)[apb_name]]
    min_row = mep_frame.iloc[mep_cropped.idxmin(axis=0)[apb_name]]
    mep_size = max_row[apb_name] - min_row[apb_name]
    return mep_size, min_row, max_row

def crop_wavelet(wt, start_freq, end_freq, start_time_millis, end_time_millis):
    wt = wt[(wt.index >= start_freq) * (wt.index <= end_freq)]
    wt = wt.loc[:, wt.columns.isin(wt.columns[(wt.columns >= (start_time_millis/1000)) * (wt.columns <= (end_time_millis / 1000))])]
    return wt

#%%
features_filename = '55-features-v2.xlsx'
wavelet_path = 'wavelets-100/'

#%%
df = pd.read_excel(features_filename, index_col=0)

p1 = np.percentile(df['mep_category_cmap'], 50)
cat = np.ones(len(df['mep_category_cmap'])) * (df['mep_category_cmap'] > p1)
df['mep_category_cmap_across_subjects'] = cat

#%%
# -- Plot subject vs MEP size (absolute, CMAP, percentile).
# -- Plot abnormal MEPs.
# -- Record the abromal MEPs to exclude in the futre processing. sub14 has all bad MEPs.
# -- Calculate wavelet powers for all electrodes.
# Implement linear mixed models using wavelet powers (all electrodes).
# Observe performance of the mixed models.
# Calculate phase and power using paper from Sara J. Hussain.
# Create some visualizations with phase, power and MEP sizes.
# Implement mixed models using phase power (all electrodes).
# Observe mixed model performance difference between.

#%% # Plot subject vs MEP size (absolute, CMAP, percentile).
plt.figure()
ax = sns.catplot(x="sub", y="mep_size", data=df, kind = "swarm")
plt.xlabel('Subject')
plt.ylabel('MEP Size (absolute)')

#%%
df_percentile = df.groupby(['sub', 'mep_category_absolute_binary']).count()
df_percentile = df_percentile.reset_index()
sns.catplot(x='sub', y='all_all_max_amplitude', data=df_percentile, hue='mep_category_absolute_binary', kind='bar')
plt.xlabel('Subject')
plt.ylabel('Number of MEPs')

#%%
df_percentile = df.groupby(['sub', 'mep_category_percentile']).count()
df_percentile = df_percentile.reset_index()
sns.catplot(x='sub', y='all_all_max_amplitude', data=df_percentile, hue='mep_category_percentile', kind='bar')
plt.xlabel('Subject')
plt.ylabel('Number of MEPs')

#%%
df_percentile = df.groupby(['sub', 'mep_category_cmap_across_subjects']).count()
df_percentile = df_percentile.reset_index()
sns.catplot(x='sub', y='all_all_max_amplitude', data=df_percentile, hue='mep_category_cmap_across_subjects', kind='bar')
plt.xlabel('Subject')
plt.ylabel('Number of MEPs')

#%% Plot abnormal MEPs.
for eeg_path in tqdm(eegs[24:24]):
    mep_path = get_mep_path(eeg_path)
    mep_frames = open_mep_as_df(mep_path)
    i = 0
    for mep_frame in mep_frames:
        apb_name = get_apb_column_name(mep_frame)
        plt.figure(figsize=(12, 4))
        plt.plot(mep_frame['s'], mep_frame[apb_name])
        plt.xlabel('Time (s)')
        plt.ylabel('MEP Size (mV)')
        plt.title(eeg_path + ' - ' + str(i))

        _, time = crop_mep_region(mep_frame)
        _, max_row, min_row = calculate_mep_size(mep_frame)
        plt.axvspan(time['s'].head(1).values[0], time['s'].tail(1).values[0], color='red', alpha=0.4)
        plt.plot(max_row['s'], max_row[apb_name], 'o', color='white')
        plt.plot(min_row['s'], min_row[apb_name], 'o', color='white')

        i = i+1

#%% Record the abromal MEPs to exclude in the futre processing. sub14 has all bad MEPs.
eegs = [
    'data/original/sub03/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub03/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub03/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
#     'data/original/sub03/exp02/eeg/SP 110RMT r1/clean-prestimulus.set', NO CMAP
#     'data/original/sub03/exp02/eeg/SP 110RMT r2/clean-prestimulus.set', NO CMAP
#     'data/original/sub03/exp03/eeg/SP 110RMT r1/clean-prestimulus.set', NO CMAP
    'data/original/sub03/exp03/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub03/exp03/eeg/SP 110RMT r3/clean-prestimulus.set',
#     'data/original/sub04/exp01/eeg/SP 110RMT r1/clean-prestimulus.set', NO CMAP
#     'data/original/sub04/exp01/eeg/SP 110RMT r2/clean-prestimulus.set', NO CMAP
#     'data/original/sub04/exp01/eeg/SP 110RMT r3/clean-prestimulus.set', NO CMAP
#     'data/original/sub05/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub06/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub06/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub06/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub06/exp02/eeg/SP 110RMT/clean-prestimulus.set',
    'data/original/sub07/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub07/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub07/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub08/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub08/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub08/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub08/exp02/eeg/SP 110RMT/clean-prestimulus.set',
    'data/original/sub08/exp03/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub08/exp03/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub08/exp03/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub12/exp02/eeg/SP 110RMT/clean-prestimulus.set',
#     'data/original/sub13/exp01/eeg/SP 110RMT/clean-prestimulus.set', LEFT HANDED
#     'data/original/sub14/exp01/eeg/SP 110RMT r1/clean-prestimulus.set', BAD MEPs
    'data/original/sub15/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub15/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub15/exp01/eeg/SP 110RMT r3/clean-prestimulus.set',
    'data/original/sub16/exp01/eeg/SP 110RMT r1/clean-prestimulus.set',
    'data/original/sub16/exp01/eeg/SP 110RMT r2/clean-prestimulus.set',
    'data/original/sub16/exp01/eeg/SP 110RMT r3/clean-prestimulus.set']

df2 = df[df['sub'] != 'sub14']
df = df2

#%%
time = pickle.load(open("90-wt_time.p", "rb")) / 1000
freq = pickle.load(open("90-wt_freq.p", "rb")) * 1000
channels = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']

#%% Calculate wavelet powers for all electrodes.
def get_powers(epoch_row):
    path = 'wavelets-100/' + epoch_row['sub'] + '-' + epoch_row['exp'] + '-' + epoch_row['run'] + '-' + str(epoch_row['epoch']) + '.pickle'
    wt = read_wavelets(path)
    powers = {
        'sub': epoch_row['sub'],
        'exp': epoch_row['exp'],
        'run': epoch_row['run'],
        'epoch': epoch_row['epoch'],
        'mep_size': epoch_row['mep_size'],
        'mep_category_absolute': epoch_row['mep_category_absolute'],
        'mep_category_absolute_binary': epoch_row['mep_category_absolute_binary'],
        'mep_category_binary': epoch_row['mep_category_binary'],
        'mep_category_cmap': epoch_row['mep_category_cmap'],
        'mep_category_cmap_across_subjects': epoch_row['mep_category_cmap_across_subjects'],
        'mep_category_cmap_category': epoch_row['mep_category_cmap_category'],
        'mep_category_cmap_category_binary': epoch_row['mep_category_cmap_category_binary'],
        'mep_category_percentile': epoch_row['mep_category_percentile']
    }
    for i in range(wt.shape[2]):
        channel = channels[i]
        df_wt = pd.DataFrame(data=wt[:, :, i], columns=time)
        df_wt.insert(0, 'frequency', freq)
        df_wt = df_wt.set_index('frequency')
        df_wt_alpha = crop_wavelet(df_wt, 8, 13, -152, -2)
        df_wt_beta = crop_wavelet(df_wt, 13, 30, -152, -2)
        df_wt_gamma = crop_wavelet(df_wt, 30, 80, -152, -2)
        df_wt_beta_gamma = crop_wavelet(df_wt, 13, 80, -152, -2)
        powers['power_alpha_' + channel] = df_wt_alpha.values.flatten().mean()
        powers['power_beta_' + channel] = df_wt_beta.values.flatten().mean()
        powers['power_gamma_' + channel] = df_wt_gamma.values.flatten().mean()
        powers['power_beta_gamma_' + channel] = df_wt_beta_gamma.values.flatten().mean()
        powers['power_b2a_' + channel] = powers['power_beta_' + channel] / powers['power_alpha_' + channel]
        powers['power_g2a_' + channel] = powers['power_gamma_' + channel] / powers['power_alpha_' + channel]
        powers['power_bg2a_' + channel] = powers['power_beta_gamma_' + channel] / powers['power_alpha_' + channel]
    return powers

num_cores = multiprocessing.cpu_count()
powers_list = Parallel(n_jobs=num_cores)(delayed(get_powers)(epoch) for idx, epoch in tqdm(df.iterrows(), total=df.shape[0]))

# powers_list = []
# for i in powers:
#     for j in i:
#         powers_list.append(j)
df_powers = pd.DataFrame(powers_list)
df_powers.head()

#%% Visualize powers.
plt.figure()
sns.distplot(df_powers[df_powers['mep_category_absolute_binary'] == 0]['power_beta_C3'], bins=10, rug=True)
plt.title('Beta Power Histogram of Small MEP')

plt.figure()
sns.distplot(df_powers[df_powers['mep_category_absolute_binary'] == 1]['power_beta_C3'], bins=10, rug=True)
plt.title('Beta Power Histogram of Large MEP')

plt.figure()
sns.distplot(df_powers[df_powers['mep_category_absolute_binary'] == 0]['power_alpha_C3'], bins=10, rug=True)
plt.title('Alpha Power Histogram of Small MEP')

plt.figure()
sns.distplot(df_powers[df_powers['mep_category_absolute_binary'] == 1]['power_alpha_C3'], bins=10, rug=True)
plt.title('Alpha Power Histogram of Large MEP')

plt.figure()
sns.distplot(df_powers[df_powers['mep_category_absolute_binary'] == 0]['power_gamma_C3'], bins=10, rug=True)
plt.title('Gamma Power Histogram of Small MEP')

plt.figure()
sns.distplot(df_powers[df_powers['mep_category_absolute_binary'] == 1]['power_gamma_C3'], bins=10, rug=True)
plt.title('Gamma Power Histogram of Large MEP')

plt.figure()
sns.distplot(df_powers[df_powers['mep_category_absolute_binary'] == 0]['power_b2a_C3'], bins=10, rug=True)
plt.title('Beta/Alpha Power Histogram of Small MEP')

plt.figure()
sns.distplot(df_powers[df_powers['mep_category_absolute_binary'] == 1]['power_b2a_C3'], bins=10, rug=True)
plt.title('Beta/Alpha Power Histogram of Large MEP')

#%% Implement linear models using wavelet powers (all electrodes).
accuracy_list = []
x_train, x_test, y_train, y_test = train_test_split(df_powers.drop([
    'epoch',
    'exp',
    'run',
    'sub',
    'mep_category_absolute',
    'mep_category_absolute_binary',
    'mep_category_binary',
    'mep_category_cmap',
    'mep_category_cmap_across_subjects',
    'mep_category_cmap_category',
    'mep_category_cmap_category_binary',
    'mep_category_percentile',
    'mep_size'
], axis=1), df_powers['mep_category_absolute_binary'])
model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5
acc = accuracy_score(y_test, y_pred)
acc = {'sub': 'all', 'accuracy': acc, 'r2': model.score(x_test, y_test), 'model': 'Linear Regression'}
accuracy_list.append(acc)
print(acc)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('Linear Regression')

#%% Decision Tree
model = DecisionTreeClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5
acc = accuracy_score(y_test, y_pred)
acc = {'sub': 'all', 'accuracy': acc, 'r2': model.score(x_test, y_test), 'model': 'Decision Tree'}
accuracy_list.append(acc)
print(acc)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('Decision Tree')

#%% SGD
model = SGDClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5
acc = accuracy_score(y_test, y_pred)
acc = {'sub': 'all', 'accuracy': acc, 'r2': model.score(x_test, y_test), 'model': 'SGD'}
accuracy_list.append(acc)
print(acc)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('SGD')

#%% Gradient Boosting
model = GradientBoostingClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5
acc = accuracy_score(y_test, y_pred)
acc = {'sub': 'all', 'accuracy': acc, 'r2': model.score(x_test, y_test), 'model': 'GradientBoostingClassifier'}
accuracy_list.append(acc)
print(acc)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('GradientBoostingClassifier')

#%% Random Forest
model = RandomForestClassifier().fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5
acc = accuracy_score(y_test, y_pred)
acc = {'sub': 'all', 'accuracy': acc, 'r2': model.score(x_test, y_test), 'model': 'Random Forest'}
accuracy_list.append(acc)
print(acc)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('Random Forest')

#%% Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=x_test.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#%% Correlation map
corrmat = df_powers.drop(['epoch', 'exp', 'run', 'sub'], axis=1).corr()
top_features = corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(df_powers[top_features].corr(), annot=True)

#%% Feature Importance


#%% Implement linear models including all subjects.
subs = df_powers['sub'].unique()
df_powers2 = pd.concat([df_powers, pd.get_dummies(df_powers['sub'])], axis=1)
x_train, x_test, y_train, y_test = train_test_split(df_powers2.drop([
    'epoch',
    'exp',
    'run',
    'sub',
    'mep_category_absolute',
    'mep_category_absolute_binary',
    'mep_category_binary',
    'mep_category_cmap',
    'mep_category_cmap_across_subjects',
    'mep_category_cmap_category',
    'mep_category_cmap_category_binary',
    'mep_category_percentile',
    'mep_size'
], axis=1), df_powers2['mep_category_absolute_binary'])
model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5
acc = accuracy_score(y_test, y_pred)
acc = {'sub': 'all including sub variable', 'accuracy': acc, 'r2': model.score(x_test, y_test)}
accuracy_list.append(acc)
print(acc)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.xlabel('Target')
plt.ylabel('Prediction')



#%%
md = smf.mixedlm("mep_category_absolute_binary ~ power_beta", df_powers[['power_alpha', 'power_beta', 'power_gamma', 'power_beta_gamma', 'power_b2a', 'power_g2a', 'power_bg2a', 'sub', 'mep_category_absolute_binary']], groups=df_powers["sub"])
mdf = md.fit()
print(mdf.summary())

#%%
