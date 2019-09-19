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
fft_path = 'fft-97/'

#%%
df = pd.read_excel(features_filename, index_col=0)

p1 = np.percentile(df['mep_category_cmap'], 50)
cat = np.ones(len(df['mep_category_cmap'])) * (df['mep_category_cmap'] > p1)
df['mep_category_cmap_across_subjects'] = cat

#%%
def read_ffts(fft_path, epoch):
    path = fft_path + str(epoch['sub']) + '-' + str(epoch['exp']) + '-' + str(epoch['run']) + '-' + str(epoch['epoch']+1) + '.pickle'
    df_fft = pickle.load(open(path, "rb"))
    df_fft = df_fft[(df_fft.index >= 8) & (df_fft.index < 70)]
    df_fft.loc[:, :] = np.real(df_fft.values)
    fft = df_fft.transpose().values
    return np.reshape(fft, (fft.shape[0], fft.shape[1], 1))

#%%
num_cores = multiprocessing.cpu_count()
x = Parallel(n_jobs=num_cores)(delayed(read_ffts)(fft_path, epoch) for idx, epoch in tqdm(df.iterrows(), total=df.shape[0]))
x = np.array(x)

#%%
y = df['mep_category_absolute_binary'].values

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=28)

#%% 
random_state = 68
run_name = "run-gpu1"
logdir = 'logs/tensorboard/103-fft/'

#%%
def train_test_model(logdir, hparams):
    filter_kernel_2 = json.loads(hparams['filter_kernel_2'])
    units = json.loads(hparams['units'])
    kernel_1 = json.loads(hparams['kernel_1'])

    # C3 Convolution.
    fft_input = keras.Input(shape=(x_train[0].shape[0], x_train[0].shape[1], 1))
    model = layers.Conv2D(filters=int(hparams['filter_1']), kernel_size=(int(kernel_1[0]), int(kernel_1[1])), activation='relu')(fft_input)
    model = layers.MaxPooling2D(pool_size=2)(model)
    model = layers.Dropout(hparams['dropout'])(model)
    if int(filter_kernel_2[0]) > 0:
        model = layers.Conv2D(filters=int(filter_kernel_2[0]), kernel_size=(1, int(filter_kernel_2[1])), activation='relu')(model)
        model = layers.MaxPooling2D(pool_size=2)(model)
        model = layers.Dropout(hparams['dropout'])(model)
    model = layers.Flatten()(model)

    # Dense concatenation.
    for unit in units:
        model = layers.Dense(unit, activation='relu')(model)
        model = layers.Dropout(hparams['dropout'])(model)
    model = layers.Dense(1, activation='sigmoid')(model)

    model = tf.keras.models.Model(inputs=fft_input, outputs=model)
    model.summary()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    plot_model(model, logdir + '/model.png', show_shapes=True, show_layer_names=False)
    model.compile(optimizer=optimizers.Adam(learning_rate=hparams['lr'], decay=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    cb = [
        callbacks.TensorBoard(log_dir=logdir),
        hp.KerasCallback(logdir, hparams)
    ]

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=2000, callbacks=cb, verbose=0)
    return model, history


#%%
def get_randomized_hyperparams():
    df_params = []
    for units in ['[128]']:
        for dropout in [0.3]:
            for lr in [0.01]:
                for kernel_1 in ['[1, 10]']:
                    for filter_1 in [8, 32]:
                        for filter_kernel_2 in ['[0, 0]']:
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
session = 1
params = []
models = []
histories = []
for row in tqdm(df_hparams.iterrows(), total=df_hparams.shape[0]):
    params.append(row)
    model, history = train_test_model(logdir + run_name + '-' + str(session), row[1])
    models.append(model)
    histories.append(history)
    session = session + 1

#%%
hist = histories[0]
plt.figure(figsize=(12, 4))
plt.plot(hist.history['val_accuracy'])
plt.title('Val accuracy')
plt.figure(figsize=(12, 4))
plt.plot(hist.history['accuracy'])
plt.title('Train accuracy')
plt.figure(figsize=(12, 4))
plt.plot(hist.history['val_loss'])
plt.title('Val loss')
plt.figure(figsize=(12, 4))
plt.plot(hist.history['loss'])
plt.title('Train loss')

#%%
hist = histories[1]
plt.figure(figsize=(12, 4))
plt.plot(hist.history['val_accuracy'])
plt.title('Val accuracy')
plt.figure(figsize=(12, 4))
plt.plot(hist.history['accuracy'])
plt.title('Train accuracy')
plt.figure(figsize=(12, 4))
plt.plot(hist.history['val_loss'])
plt.title('Val loss')
plt.figure(figsize=(12, 4))
plt.plot(hist.history['loss'])
plt.title('Train loss')

#%%
