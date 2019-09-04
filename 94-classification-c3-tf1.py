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
from tensorflow.keras import layers
import timeit
from skimage.transform import resize
from timeit import default_timer as timer
from datetime import timedelta
import json

import keras
from keras import layers
from keras.models import Model
from keras import optimizers
from keras import callbacks
from keras.utils import plot_model
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
x = []
for wt in tqdm(df_wt_c3):
    img = np.array(wt.values)
    img = resize(img, (160, 160))
    x.append(img.reshape(img.shape[0], img.shape[1], 1))
    
x = np.array(x)
y = cat.values

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=28)

#%%
def train_test_model(logdir, hparams):
    filter_kernel_2 = json.loads(hparams['filter_kernel_2'])
    
    inputs = keras.Input(shape=(x_train[0].shape[0], x_train[0].shape[1], 1), name='c3_input')
    model = layers.Conv2D(filters=int(hparams['filter_1']), kernel_size=int(hparams['kernel_1']), activation='relu')(inputs)
    model = layers.MaxPooling2D(pool_size=2)(model)
    model = layers.Dropout(0.4)(model)
    if int(filter_kernel_2[0]) > 0:
        model = layers.Conv2D(filters=int(filter_kernel_2[0]), kernel_size=int(filter_kernel_2[1]), activation='relu')(model)
        model = layers.MaxPooling2D(pool_size=2)(model)
        model = layers.Dropout(0.4)(model)
    model = layers.GlobalAvgPool2D()(model)
    model = layers.Dense(hparams['units_1'], activation='relu')(model)
    if int(hparams['units_2']) > 0:
        model = layers.Dense(int(hparams['units_2']), activation='relu')(model)
    model = layers.Dense(1, activation='sigmoid')(model)
    model = Model(inputs=inputs, outputs=model, name='c3_model')

    model.compile(optimizer=optimizers.Adam(lr=hparams['lr'], decay=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    cb = [
        callbacks.TensorBoard(log_dir=logdir)
    ]
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1000, callbacks=cb, verbose=0)
    return model, history

#%%
run_name = "run-c3"
logdir = 'logs/tensorboard/94-c3-c4-v1/'
model, history = train_test_model(logdir + run_name, {'units_1': 128, 'units_2': 16, 'lr': 0.0001, 'kernel_1': 5, 'filter_1': 8, 'filter_kernel_2': '[8, 10]'})

#%%
model.save(logdir + 'c3.h5')