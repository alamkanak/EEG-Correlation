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
import tensorflow as tf
from tensorflow.keras import layers
import timeit
from skimage.transform import resize
from timeit import default_timer as timer
from datetime import timedelta
import json
from tensorflow.keras.utils import plot_model

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

#%%
features_filename = '55-features-v1.xlsx'


#%%
df = pd.read_excel(features_filename, index_col=0)
p1 = np.percentile(df['mep_category_cmap'], 50)
cat = np.ones(len(df['mep_category_cmap'])) * (df['mep_category_cmap'] > p1)
df['mep_category_cmap_across_subjects'] = cat


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
    wt_ltm1 = wt_ltm1[(wt_ltm1.index * 1000 > 2) * (wt_ltm1.index * 1000 < 50)]
    wt_ltm1 = wt_ltm1.loc[:, wt_ltm1.columns.isin(wt_ltm1.columns[(wt_ltm1.columns >= (start_time_sec/1000)) * (wt_ltm1.columns <= (end_time_sec/1000))])]
    norm = normalize_image(wt_ltm1)
    df_wt.append(wt_ltm1)
    #df_mean = np.add(df_mean, norm.values)


#%%
df_wt[5].shape


#%%
# df_mean = df_mean / df.shape[0]


#%%
# df_wt2 = []
# for wt in tqdm(df_wt):
#     df_wt2.append(normalize_image(np.add(wt, -df_mean)))
# df_wt = df_wt2


#%%
x = []
for wt in df_wt:
    img = np.array(wt.values)
    img = resize(img, (160, 160))
    x.append(img.reshape(img.shape[0], img.shape[1], 1))
    
x = np.array(x)
y = df['mep_category_absolute_binary'].values

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=28)

#%%
random_state = 68
logdir = 'logs/tensorboard/98-small-v2/'
run_name = 'run1'

#%%
def train_test_model(logdir, hparams):
    classifier = tf.keras.Sequential()
    classifier.add(tf.keras.layers.Conv2D(filters=hparams['filter_1'], kernel_size=hparams['kernel_1'], activation='relu', input_shape=(x_train[0].shape[0], x_train[0].shape[1], 1)))
    if hparams['batch_norm']:
        classifier.add(tf.keras.layers.BatchNormalization())
    classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    classifier.add(tf.keras.layers.Dropout(hparams['dropout']))
    classifier.add(tf.keras.layers.Conv2D(filters=hparams['filter_2'], kernel_size=hparams['kernel_2'], activation='relu'))
    classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    classifier.add(tf.keras.layers.Dropout(hparams['dropout']))
    classifier.add(tf.keras.layers.Flatten())
    classifier.add(tf.keras.layers.Dense(hparams['units'], activation='relu'))
    classifier.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    plot_model(classifier, logdir + '/model.png', show_shapes=True, show_layer_names=False)
    
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['lr']), loss='binary_crossentropy', metrics=['accuracy'])

    cb = [
        tf.keras.callbacks.TensorBoard(log_dir=logdir),
        hp.KerasCallback(logdir, hparams)
    ]
    classifier.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=250, callbacks=cb, verbose=0)

    _, accuracy = classifier.evaluate(x_test, y_test)
    end = timer()
    return accuracy

#%%
df_params = []
for batch_norm in [True, False]:
    for dropout in [0.3, 0.5]:
        for filter_1 in [8, 32, 128]:
            for filter_2 in [8, 32, 128]:
                for num_units in [64, 128, 256]:
                    for kernel_1 in [5, 10, 25]:
                        for kernel_2 in [5, 10, 25]:
                            for lr in [0.0001, 0.001, 0.00001]:
                                    hparams = {
                                        'num_units': num_units,
                                        'dropout': dropout,
                                        'lr': lr,
                                        'kernel_1': kernel_1,
                                        'kernel_2': kernel_2,
                                        'filter_1': filter_1,
                                        'filter_2': filter_2,
                                        'batch_norm': batch_norm
                                    }
                                    df_params.append(hparams)
df_params = pd.DataFrame(df_params)
df_params = df_params.sample(frac=1, random_state=random_state)
df_params.head()

#%%
df_params.shape

#%%
session_num = 0
for idx, row in tqdm(df_params.iterrows(), total=df_params.shape[0]):
    hparams = {
        'units': row['num_units'],
        'dropout': row['dropout'],
        'lr': row['lr'],
        'kernel_1': row['kernel_1'],
        'kernel_2': row['kernel_2'],
        'filter_1': row['filter_1'],
        'filter_2': row['filter_2'],
        'batch_norm': row['batch_norm']
    }
    session_name = run_name + "-%d" % session_num
    train_test_model(logdir + session_name, hparams)
    session_num += 1

#%%
