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
    path = 'wavelets-90/' + sub + '-' + exp + '-' + run + '-' + str(epoch_num)
    with open(path + '-C3.pickle', 'rb') as f:
        c3 = pickle.load(f)
    with open(path + '-C4.pickle', 'rb') as f:
        c4 = pickle.load(f)
    return c3, c4

def wavelet_band_max(df, interval):
    indices = []
    for el in (df.index * 1000):
        indices.append(el in interval)
    df = df[indices]
    if (df.shape[0] == 0):
        return 0, 0, 0, 0
    return df.mean(axis=1).max(), df.mean(axis=1).argmax() * 1000, df.mean(axis=0).max(), df.mean(axis=0).argmax()

#%%
features_filename = '55-features-v2.xlsx'


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

df_wt_c3 = []
df_wt_c4 = []
df_mean_c3 = np.zeros((37,983))
df_mean_c4 = np.zeros((37,983))
for idx, epoch in tqdm(df.iterrows(), total=df.shape[0]):
    wt_c3, wt_c4 = read_wavelets(epoch['sub'], epoch['exp'], epoch['run'], epoch['epoch'])
    wt_c3 = wt_c3[(wt_c3.index * 1000 > 2) * (wt_c3.index * 1000 < 50)]
    wt_c3 = wt_c3.loc[:, wt_c3.columns.isin(wt_c3.columns[(wt_c3.columns >= (start_time_sec/1000)) * (wt_c3.columns <= (end_time_sec/1000))])]
    norm_c3 = normalize_image(wt_c3)
    df_wt_c3.append(norm_c3)

    wt_c4 = wt_c4[(wt_c4.index * 1000 > 2) * (wt_c4.index * 1000 < 50)]
    wt_c4 = wt_c4.loc[:, wt_c4.columns.isin(wt_c4.columns[(wt_c4.columns >= (start_time_sec/1000)) * (wt_c4.columns <= (end_time_sec/1000))])]
    norm_c4 = normalize_image(wt_c4)
    df_wt_c4.append(norm_c4)

    df_mean_c3 = np.add(df_mean_c3, norm_c3.values)
    df_mean_c4 = np.add(df_mean_c4, norm_c4.values)


#%%
df_wt_c3[5].shape


#%%
df_mean_c3 = df_mean_c3 / df.shape[0]
df_mean_c4 = df_mean_c4 / df.shape[0]


#%% Subtract mean
df_wt_c3_2 = []
for wt in tqdm(df_wt_c3):
    df_wt_c3_2.append(normalize_image(np.add(wt, -df_mean_c3)))
df_wt_c3 = df_wt_c3_2

df_wt_c4_2 = []
for wt in tqdm(df_wt_c4):
    df_wt_c4_2.append(normalize_image(np.add(wt, -df_mean_c4)))
df_wt_c4 = df_wt_c4_2

#%%
x_c3 = []
for wt in tqdm(df_wt_c3):
    img = np.array(wt.values)
    img = resize(img, (160, 160))
    x_c3.append(img.reshape(img.shape[0], img.shape[1], 1))
x_c4 = []
for wt in tqdm(df_wt_c4):
    img = np.array(wt.values)
    img = resize(img, (160, 160))
    x_c4.append(img.reshape(img.shape[0], img.shape[1], 1))

x = []
for i in range(len(x_c3)):
    x.append((np.array(x_c3[i]), np.array(x_c4[i])))
y = cat.values

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=28)
x_train_c3 = []
x_test_c3 = []
x_train_c4 = []
x_test_c4 = []
for item in x_train:
    x_train_c3.append(item[0])
    x_train_c4.append(item[1])
for item in x_test:
    x_test_c3.append(item[0])
    x_test_c4.append(item[1])

#%% 
model_path = 'logs/tensorboard/93-c3-c4-v1/'
random_state = 64
run_name = "run-c3-c4-1"
logdir = 'logs/tensorboard/95-c3-c4-v1/'

#%%
# tf.keras.backend.clear_session()
# model_c3 = load_model(model_path + 'c3.h5')
# model_c4 = load_model(model_path + 'c4.h5')

#%%
def train_test_model(logdir, hparams):
    filter_kernel_2 = json.loads(hparams['filter_kernel_2'])

    # C3 Convolution.
    c3_input = keras.Input(shape=(x_train_c3[0].shape[0], x_train_c3[0].shape[1], 1), name='c3_input')
    c3_model = layers.Conv2D(filters=int(hparams['filter_1']), kernel_size=int(hparams['kernel_1']), activation='relu')(c3_input)
    c3_model = layers.MaxPooling2D(pool_size=2)(c3_model)
    c3_model = layers.Dropout(0.4)(c3_model)
    if int(filter_kernel_2[0]) > 0:
        c3_model = layers.Conv2D(filters=int(filter_kernel_2[0]), kernel_size=int(filter_kernel_2[1]), activation='relu')(c3_model)
        c3_model = layers.MaxPooling2D(pool_size=2)(c3_model)
        c3_model = layers.Dropout(0.4)(c3_model)
    c3_model = layers.Flatten()(c3_model)

    # C4 Convolution.
    c4_input = keras.Input(shape=(x_train_c3[0].shape[0], x_train_c3[0].shape[1], 1), name='c4_input')
    c4_model = layers.Conv2D(filters=int(hparams['filter_1']), kernel_size=int(hparams['kernel_1']), activation='relu')(c4_input)
    c4_model = layers.MaxPooling2D(pool_size=2)(c4_model)
    c4_model = layers.Dropout(0.4)(c4_model)
    if int(filter_kernel_2[0]) > 0:
        c4_model = layers.Conv2D(filters=int(filter_kernel_2[0]), kernel_size=int(filter_kernel_2[1]), activation='relu')(c4_model)
        c4_model = layers.MaxPooling2D(pool_size=2)(c4_model)
        c4_model = layers.Dropout(0.4)(c4_model)
    c4_model = layers.Flatten()(c4_model)

    # Dense concatenation.
    model = layers.Concatenate()([c3_model, c4_model])
    for unit in hparams['units']:
        model = layers.Dense(unit, activation='relu')(model)
        model = layers.Dropout(hparams['dropout'])(model)
    model = layers.Dense(1, activation='sigmoid')(model)

    model = tf.keras.models.Model(inputs=[c3_input, c4_input], outputs=model)
    model.summary()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    plot_model(model, logdir + '/model.png', show_shapes=True, show_layer_names=False)
    model.compile(optimizer=optimizers.Adam(learning_rate=hparams['lr'], decay=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # cb = [
    #     callbacks.TensorBoard(log_dir=logdir),
    #     hp.KerasCallback(logdir, hparams)
    # ]

    # plot_model('model.png')
    # history = classifier.fit({'c3_input': x_train_c3, 'c4_input': x_train_c4}, y_train, validation_data=({'c3_input': x_test_c3, 'c4_input': x_test_c4}, y_test), batch_size=64, epochs=1000, callbacks=cb, verbose=0)
    # return classifier, history


#%%
def get_randomized_hyperparams():
    df_params = []
    for units in [[128, 64], [128], [64, 8], [32]]:
        for dropout in [0.3, 0.5]:
            for lr in [0.0001, 0.00001, 0.01]:
                for kernel_1 in [5]:
                    for filter_1 in [8]:
                        for filter_kernel_2 in ['[8, 10]']:
                            hparams = {
                                'units': units,
                                'dropout': dropout,
                                'lr': lr,
                                'kernel_1': kernel_1,
                                'filter_1': filter_1,
                                'filter_kernel_2': filter_kernel_2
                            }
                            df_params.append(hparams)
    df_params = pd.DataFrame(df_params)
    df_params = df_params.sample(frac=1, random_state=random_state)
    print('Total hyperparameter combinations: ' + str(df_params.shape))
    return df_params

#%%
df_hparams = get_randomized_hyperparams()
df_hparams.head()

#%%
for row in tqdm(df_hparams.iterrows(), total=df_hparams.shape[0]):
    train_test_model(logdir + run_name, row[1])
    break

#%%
