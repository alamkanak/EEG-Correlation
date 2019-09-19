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

#%%
features_filename = '55-features-v2.xlsx'
wavelet_path = 'wavelets-100/'

#%%
df = pd.read_excel(features_filename, index_col=0)

p1 = np.percentile(df['mep_category_cmap'], 50)
cat = np.ones(len(df['mep_category_cmap'])) * (df['mep_category_cmap'] > p1)
df['mep_category_cmap_across_subjects'] = cat


#%%
files = sorted(glob.glob(wavelet_path + '*.pickle'))
files_train, files_test, _, _ = train_test_split(files, files, test_size=0.25, random_state=28)

def input_generator(df, files):
    for file_path in files:
        wt = read_wavelets(file_path)
        im_size = [64, 64, 64]
        wt = scipy.ndimage.interpolation.zoom(wt, [im_size[0]/wt.shape[0], im_size[1]/wt.shape[1], im_size[2]/wt.shape[2]], order=5, prefilter=False)
        parts = file_path.split('/')[1].split('-')
        label = df[(df['sub'] == parts[0]) & (df['exp'] == parts[1]) & (df['run'] == parts[2]) & (df['epoch'] == int(parts[3].split('.')[0]))]['mep_category_absolute_binary']
        yield np.reshape(wt, (73, 1024, 64, 1)), label

train_dataset = tf.data.Dataset.from_generator(lambda: input_generator(df, files_train), output_types=(tf.float32, tf.int16), output_shapes=((73, 1024, 64, 1), (1,))).batch(32)
test_dataset = tf.data.Dataset.from_generator(lambda: input_generator(df, files_test), output_types=(tf.float32, tf.int16), output_shapes=((73, 1024, 64, 1), (1,))).batch(32)

#%% 
random_state = 68
run_name = "run-gpu1"
logdir = 'logs/tensorboard/101-wt/'

#%%
def train_test_model(logdir, hparams):
    kernel_1 = json.loads(hparams['kernel_1'])
    kernel_2 = json.loads(hparams['kernel_2'])
    units = json.loads(hparams['units'])

    wt_input = keras.Input(shape=(73, 1024, 64, 1))

    model = layers.Conv3D(filters=int(hparams['filter_1']), kernel_size=(int(kernel_1[0]), int(kernel_1[1]), int(kernel_1[2])), activation='relu')(wt_input)
    model = layers.MaxPooling3D(pool_size=(2, 2, 1))(model)
    model = layers.Dropout(hparams['dropout'])(model)
    
    model = layers.Conv3D(filters=int(hparams['filter_2']), kernel_size=(int(kernel_2[0]), int(kernel_2[1]), int(kernel_2[2])), activation='relu')(model)
    model = layers.MaxPooling3D(pool_size=(2, 2, 1))(model)
    model = layers.Dropout(hparams['dropout'])(model)

    model = layers.Flatten()(model)

    # Dense concatenation.
    for unit in units:
        model = layers.Dense(unit, activation='relu')(model)
        model = layers.Dropout(hparams['dropout'])(model)
    model = layers.Dense(1, activation='sigmoid')(model)

    model = tf.keras.models.Model(inputs=wt_input, outputs=model)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    plot_model(model, logdir + '/model.png', show_shapes=True, show_layer_names=False)
    model.compile(optimizer=optimizers.Adam(learning_rate=hparams['lr'], decay=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    cb = [
        callbacks.TensorBoard(log_dir=logdir),
        hp.KerasCallback(logdir, hparams)
    ]

    history = model.fit(train_dataset, validation_data=test_dataset, epochs=100, callbacks=cb, verbose=0)
    return model, history


#%%
def get_randomized_hyperparams():
    df_params = []
    for units in ['[128, 64]', '[128]', '[256]', '[64, 8]', '[32]']:
        for dropout in [0.3, 0.5]:
            for lr in [0.0001, 0.00001, 0.01]:
                for kernel_1 in ['[5, 5, 1]', '[10, 10, 1]', '[3, 3, 1]', '[5, 5, 5]', '[10, 10, 10]']:
                    for filter_1 in [8, 32, 128]:
                        for filter_2 in [8, 32, 128]:
                            for kernel_2 in ['[5, 5, 1]', '[10, 10, 1]', '[3, 3, 1]', '[5, 5, 5]', '[10, 10, 10]']:
                                hparams = {
                                    'units': units,
                                    'dropout': dropout,
                                    'lr': lr,
                                    'kernel_1': kernel_1,
                                    'filter_1': filter_1,
                                    'kernel_2': kernel_2,
                                    'filter_2': filter_2
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
for row in tqdm(df_hparams.iterrows(), total=df_hparams.shape[0]):
    train_test_model(logdir + run_name + '-' + str(session), row[1])
    session = session + 1
