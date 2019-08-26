# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

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
from tqdm import tqdm, tqdm_notebook

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from keras_tqdm import TQDMNotebookCallback
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
import tensorflow as tf
from tensorflow.keras import layers
import timeit
import os

#%%
eeglab_path = '/home/raquib/Documents/MATLAB/eeglab2019_0/functions/'
octave.addpath(eeglab_path + 'guifunc');
octave.addpath(eeglab_path + 'popfunc');
octave.addpath(eeglab_path + 'adminfunc');
octave.addpath(eeglab_path + 'sigprocfunc');
octave.addpath(eeglab_path + 'miscfunc');


#%%
experiment = 'data/original/*/*'
meps = sorted(glob.glob(experiment + '/mep/*/*.txt'))
mep_present = len(meps) > 0
eegs = sorted(glob.glob(experiment + '/eeg/*/clean-prestimulus.set'))
eeg_present = len(eegs) > 0
cmaps = sorted(glob.glob(experiment + '/cmap/*.xlsx'))
cmap_present = len(cmaps) > 0
all_present = mep_present and eeg_present and cmap_present
print(all_present)


#%%
print('EEG count: ' + str(len(eegs)))
print('MEP count: ' + str(len(meps)))
print('CMAP count: ' + str(len(cmaps)))


#%%
eegs


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
    'data/original/sub13/exp01/eeg/SP 110RMT/clean-prestimulus.set',
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

# Calculate EEG area.
def calculate_eeg_area(epoch_df, sf=2048):
    y = epoch_df.drop('time', axis=1).mean(axis=1)
    b2, a2 = signal.butter(4, 200/(sf/2), btype='lowpass')
    envelope = signal.filtfilt(b2, a2, np.abs(y))
    area = np.trapz(envelope, epoch_df['time'].values)
    return area

# Calculate EEG frequency.
def calculate_eeg_frequency(channel):
    sf = 2048
    win = 4 * sf
    freqs, psd = signal.welch(channel, sf, nperseg=win)
    return freqs, psd

def calculate_eeg_max_amplitude(epoch_df):
    avg = epoch_df.mean(axis=1)
    return np.max(avg.values)

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

def filter_electrodes(trial, which='all'):
    time_column = trial['time']
    if which == 'ltm1':
        channel_names = ['FC5','FC1','C3','CP5','CP1','FC3','C5','C1','CP3']
    elif which == 'rtm1':
        channel_names = ['FC6','FC2','C4','CP6','CP2','FC4','C6','C2','CP4']
    elif which == 'central':
        channel_names = ['Fz','FCz','Cz','F1','FC1','C1','C2','FC2','F2']
    else:
        channel_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz']
    trial = trial[channel_names]
    trial['time'] = time_column
    return trial

def read_wavelets(sub, exp, run, epoch_num):
    path = 'wavelets/' + sub + '-' + exp + '-' + run + '-' + str(epoch_num)
    with open(path + '-central.pickle', 'rb') as f:
        central = pickle.load(f)
    with open(path + '-ltm1.pickle', 'rb') as f:
        ltm1 = pickle.load(f)
    with open(path + '-rtm1.pickle', 'rb') as f:
        rtm1 = pickle.load(f)
    with open(path + '-all.pickle', 'rb') as f:
        all_channels = pickle.load(f)
    return all_channels, ltm1, rtm1, central

def wavelet_band_max(df, interval):
    indices = []
    for el in (df.index * 1000):
        indices.append(el in interval)
    df = df[indices]
    if (df.shape[0] == 0):
        return 0, 0, 0, 0
    return df.mean(axis=1).max(), df.mean(axis=1).argmax() * 1000, df.mean(axis=0).max(), df.mean(axis=0).argmax()

#%% [markdown]
# # Read features file

#%%
features_filename = '55-features-v1.xlsx'


#%%
df = pd.read_excel(features_filename, index_col=0)

p1 = np.percentile(df['mep_category_cmap'], 50)
cat = np.ones(len(df['mep_category_cmap'])) * (df['mep_category_cmap'] > p1)
df['mep_category_cmap_across_subjects'] = cat

#%% [markdown]
# # Prepare wavelet dataframe

#%%
start_time_sec = -100
end_time_sec = -20

wt_large_all_all, wt_large_ltm1_all, wt_large_rtm1_all, wt_large_central_all, wt_small_all_all, wt_small_ltm1_all, wt_small_rtm1_all, wt_small_central_all = np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164))
wt_large_all_all_avg, wt_large_ltm1_all_avg, wt_large_rtm1_all_avg, wt_large_central_all_avg, wt_small_all_all_avg, wt_small_ltm1_all_avg, wt_small_rtm1_all_avg, wt_small_central_all_avg = np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164)), np.zeros((52, 164))

df_wt = []
        
for idx, epoch in tqdm(df.iterrows(), total=df.shape[0]):
    wt_all, wt_ltm1, wt_rtm1, wt_central = read_wavelets(epoch['sub'], epoch['exp'], epoch['run'], epoch['epoch'])

    # Drop last 9 rows (freq 0 to 2Hz)
    wt_all = wt_all[(wt_all.index * 1000 > 6) * (wt_all.index * 1000 < 50)]
    wt_ltm1 = wt_ltm1[(wt_ltm1.index * 1000 > 6) * (wt_ltm1.index * 1000 < 50)]
    wt_rtm1 = wt_rtm1[(wt_rtm1.index * 1000 > 6) * (wt_rtm1.index * 1000 < 50)]
    wt_central = wt_central[(wt_central.index * 1000 > 6) * (wt_central.index * 1000 < 50)]

    # Take only last 100ms - 0
    wt_all = wt_all.loc[:, wt_all.columns.isin(wt_all.columns[(wt_all.columns >= (start_time_sec/1000)) * (wt_all.columns <= (end_time_sec/1000))])]
    wt_ltm1 = wt_ltm1.loc[:, wt_ltm1.columns.isin(wt_ltm1.columns[(wt_ltm1.columns >= (start_time_sec/1000)) * (wt_ltm1.columns <= (end_time_sec/1000))])]
    wt_rtm1 = wt_rtm1.loc[:, wt_rtm1.columns.isin(wt_rtm1.columns[(wt_rtm1.columns >= (start_time_sec/1000)) * (wt_rtm1.columns <= (end_time_sec/1000))])]
    wt_central = wt_central.loc[:, wt_central.columns.isin(wt_central.columns[(wt_central.columns >= (start_time_sec/1000)) * (wt_central.columns <= (end_time_sec/1000))])]
    
    df_wt.append(wt_ltm1)
#     val = wt_ltm1.values.flatten()
#     val = np.append(val, epoch['mep_category_cmap_across_subjects'])
#     df_wt.append(val)        

#%%
x = []
for wt in df_wt:
    x.append(wt.values.reshape(wt.shape[0], wt.shape[1], 1))
    
x = np.array(x)
y = cat


#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)


#%%
cb = TQDMNotebookCallback(show_inner=False)
cb.on_train_batch_begin = cb.on_batch_begin
cb.on_train_batch_end = cb.on_batch_end
cb.on_test_begin = cb.on_train_begin
cb.on_test_end = cb.on_train_end
cb.on_test_batch_begin = cb.on_batch_begin
cb.on_test_batch_end = cb.on_batch_end


plcb = PlotLossesCallback()
# plcb.on_train_batch_begin = plcb.on_batch_begin
# plcb.on_train_batch_end = plcb.on_batch_end
# plcb.on_test_begin = plcb.on_train_begin
# plcb.on_test_end = plcb.on_train_end
# plcb.on_test_batch_begin = plcb.on_batch_begin
# plcb.on_test_batch_end = plcb.on_batch_end

tb = tf.keras.callbacks.TensorBoard(log_dir='logs/tensorboard/60-wavelet-cnn')


#%%
start = timeit.default_timer()
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# with strategy.scope():
classifier = tf.keras.Sequential()

classifier.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(x[0].shape[0],x[0].shape[1],1))) 
classifier.add(tf.keras.layers.BatchNormalization())
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2))
classifier.add(tf.keras.layers.Dropout(0.3))

classifier.add(tf.keras.layers.Flatten())

classifier.add(tf.keras.layers.Dense(256, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.5))
classifier.add(tf.keras.layers.Dense(1, activation='sigmoid'))
classifier.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=20, callbacks=[tb], verbose=0)
stop = timeit.default_timer()
print('Time: ', stop - start)  

#%%
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
