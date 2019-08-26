# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
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
import tensorflow as tf
from tensorflow.keras import layers
import timeit
from skimage.transform import resize


#%%
eeglab_path = '/home/raquib/Documents/MATLAB/eeglab2019_0/functions/'
octave.addpath(eeglab_path + 'guifunc')
octave.addpath(eeglab_path + 'popfunc')
octave.addpath(eeglab_path + 'adminfunc')
octave.addpath(eeglab_path + 'sigprocfunc')
octave.addpath(eeglab_path + 'miscfunc')


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
def normalize_image(image, mx=1, mn=0):
    wt_min = image.min().min()
    std = (image.values - wt_min) / (image.max().max() - wt_min)
    image.loc[:, :] = std * (mx - mn) + mn
    return image


#%%
start_time_sec = -500
end_time_sec = -20

df_wt = []
df_mean = np.zeros((37,983))
for idx, epoch in tqdm(df.iterrows(), total=df.shape[0]):
    wt_all, wt_ltm1, wt_rtm1, wt_central = read_wavelets(epoch['sub'], epoch['exp'], epoch['run'], epoch['epoch'])

    # Take 6-50Hz frequencies.
#     wt_all = wt_all[(wt_all.index * 1000 > 6) * (wt_all.index * 1000 < 50)]
    wt_ltm1 = wt_ltm1[(wt_ltm1.index * 1000 > 2) * (wt_ltm1.index * 1000 < 50)]
#     wt_rtm1 = wt_rtm1[(wt_rtm1.index * 1000 > 6) * (wt_rtm1.index * 1000 < 50)]
#     wt_central = wt_central[(wt_central.index * 1000 > 6) * (wt_central.index * 1000 < 50)]

    # Take only last -100ms to -20ms.
#     wt_all = wt_all.loc[:, wt_all.columns.isin(wt_all.columns[(wt_all.columns >= (start_time_sec/1000)) * (wt_all.columns <= (end_time_sec/1000))])]
    wt_ltm1 = wt_ltm1.loc[:, wt_ltm1.columns.isin(wt_ltm1.columns[(wt_ltm1.columns >= (start_time_sec/1000)) * (wt_ltm1.columns <= (end_time_sec/1000))])]
#     wt_rtm1 = wt_rtm1.loc[:, wt_rtm1.columns.isin(wt_rtm1.columns[(wt_rtm1.columns >= (start_time_sec/1000)) * (wt_rtm1.columns <= (end_time_sec/1000))])]
#     wt_central = wt_central.loc[:, wt_central.columns.isin(wt_central.columns[(wt_central.columns >= (start_time_sec/1000)) * (wt_central.columns <= (end_time_sec/1000))])]
    
    norm = normalize_image(wt_ltm1)
    df_wt.append(norm)
    df_mean = np.add(df_mean, norm.values)

df_mean = df_mean / df.shape[0]


#%%
df_wt2 = []
for wt in tqdm(df_wt):
    df_wt2.append(normalize_image(np.add(wt, -df_mean)))
df_wt = df_wt2

#%% [markdown]
# # Keras

#%%
x = []
for wt in df_wt:
    img = np.array(wt.values)
    img = resize(img, (160, 160))
    x.append(img.reshape(img.shape[0], img.shape[1], 1))
    
x = np.array(x)
y = cat.values


#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=28)



#%%
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256, 512]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.5, 0.3]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.00001]))
HP_CNN_FILTER_1 = hp.HParam('filter_1', hp.Discrete([16, 64, 256]))
HP_CNN_FILTER_2 = hp.HParam('filter_2', hp.Discrete([16, 64, 256]))
HP_BATCH_NORM = hp.HParam('batch_norm', hp.Discrete([False, True]))
HP_CNN_KERNEL_X_1 = hp.HParam('kernel_1_x', hp.Discrete([80, 30, 5]))
HP_CNN_KERNEL_Y_1 = hp.HParam('kernel_1_y', hp.Discrete([80, 30, 5]))
HP_CNN_KERNEL_X_2 = hp.HParam('kernel_2_x', hp.Discrete([80, 30, 5]))
HP_CNN_KERNEL_Y_2 = hp.HParam('kernel_2_y', hp.Discrete([80, 30, 5]))

with tf.summary.create_file_writer('logs/78-hparam-tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_LEARNING_RATE, HP_CNN_KERNEL_X_1, HP_CNN_KERNEL_Y_1, HP_CNN_KERNEL_X_2, HP_CNN_KERNEL_Y_2, HP_CNN_FILTER_1, HP_CNN_FILTER_2, HP_BATCH_NORM],
        metrics=[hp.Metric('accuracy', display_name='Accuracy')],
    )


#%%
def train_test_model(logdir, hparams):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        classifier = tf.keras.Sequential()
        classifier.add(tf.keras.layers.Conv2D(filters=hparams[HP_CNN_FILTER_1], kernel_size=(hparams[HP_CNN_KERNEL_X_1], hparams[HP_CNN_KERNEL_Y_1]), padding='same', activation='relu', input_shape=(x_train[0].shape[0], x_train[0].shape[1],1)))
        if hparams[HP_BATCH_NORM]:
            classifier.add(tf.keras.layers.BatchNormalization())
        classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        classifier.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))
        classifier.add(tf.keras.layers.Conv2D(filters=hparams[HP_CNN_FILTER_2], kernel_size=(hparams[HP_CNN_KERNEL_X_2], hparams[HP_CNN_KERNEL_Y_2]), padding='same', activation='relu'))
        classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        classifier.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))
        classifier.add(tf.keras.layers.Flatten())
        classifier.add(tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation='relu'))
        classifier.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))
        classifier.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE], decay=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    cb = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        hp.KerasCallback(logdir, hparams)
    ]
    classifier.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=10, callbacks=cb, verbose=0)

    _, accuracy = classifier.evaluate(x_test, y_test)
    return accuracy

#%% [markdown]
# # Random search

#%%
df_params = []
for batch_norm in HP_BATCH_NORM.domain.values:
    for dropout in HP_DROPOUT.domain.values:
        for filter_1 in HP_CNN_FILTER_1.domain.values:
            for filter_2 in HP_CNN_FILTER_2.domain.values:
                for num_units in HP_NUM_UNITS.domain.values:
                    for kernel_1_x in HP_CNN_KERNEL_X_1.domain.values:
                        for kernel_1_y in HP_CNN_KERNEL_Y_1.domain.values:
                            for kernel_2_x in HP_CNN_KERNEL_X_2.domain.values:
                                for kernel_2_y in HP_CNN_KERNEL_Y_2.domain.values:
                                    for lr in HP_LEARNING_RATE.domain.values:
                                        hparams = {
                                            'num_units': num_units,
                                            'dropout': dropout,
                                            'learning_rate': lr,
                                            'kernel_1_x': kernel_1_x,
                                            'kernel_1_y': kernel_1_y,
                                            'kernel_2_x': kernel_2_x,
                                            'kernel_2_y': kernel_2_y,
                                            'filter_1': filter_1,
                                            'filter_2': filter_2,
                                            'batch_norm': batch_norm
                                        }
                                        df_params.append(hparams)
df_params = pd.DataFrame(df_params)
df_params = df_params.sample(frac=1)


#%%
session_num = 0
for idx, row in df_params.iterrows():
    hparams = {
        HP_NUM_UNITS: row['num_units'],
        HP_DROPOUT: row['dropout'],
        HP_LEARNING_RATE: row['learning_rate'],
        HP_CNN_KERNEL_X_1: row['kernel_1_x'],
        HP_CNN_KERNEL_Y_1: row['kernel_1_y'],
        HP_CNN_KERNEL_X_2: row['kernel_2_x'],
        HP_CNN_KERNEL_Y_2: row['kernel_2_y'],
        HP_CNN_FILTER_1: row['filter_1'],
        HP_CNN_FILTER_2: row['filter_2'],
        HP_BATCH_NORM: row['batch_norm']
    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    train_test_model('logs/tensorboard/78-wavelet-hyper-v1/' + run_name, hparams)
    session_num += 1

