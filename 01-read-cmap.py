#%%
import glob
import pandas as pd
import mne
import matplotlib.pyplot as plt

#%%
experiments = sorted(glob.glob('data/original/*/*'))
i = 0
for experiment in experiments:
    cmaps = sorted(glob.glob(experiment + '/cmap/*.xlsx'))
    cmap_present = len(cmaps) > 0
    meps = sorted(glob.glob(experiment + '/mep/*/*.txt'))
    mep_present = len(meps) > 0
    eegs = sorted(glob.glob(experiment + '/eeg/*/*.cnt'))
    eeg_present = len(eegs) > 0
    all_present = cmap_present and mep_present and eeg_present
    if all_present:
        i = i+1
        df = pd.read_excel(cmaps[0], sheet_name='Raw data', header=None)
        cmap = df.iloc[44:df.shape[0]-1, 38:59]
        time = df.iloc[44:df.shape[0]-1, 0]
        mean = cmap.mean(axis=1)
        mean_clean = mean.copy()
        mean_clean[0:104*10-1] = 0
        cropped = mean[104*10:130*10]
        fig = plt.figure(figsize=(14, 5))
        ax = plt.axes()
        ax.plot(time, mean)
        plt.axvspan(104, 130, color='red', alpha=0.1)
        idxmax = mean_clean.idxmax(axis=0)
        idxmax = time[idxmax]
        idxmin = mean_clean.idxmin(axis=0)
        idxmin = time[idxmin]
        max_val = mean_clean.max()
        min_val = mean_clean.min()
        ax.plot(idxmax, max_val, 'ro')
        ax.plot(idxmin, min_val, 'ro')
        ax.annotate(str(max_val), (idxmax, mean_clean.max()))
        ax.annotate(str(min_val), (idxmin, mean_clean.min()))
        ax.set_title(str(i) + '. ' + cmaps[0])
        plt.show() 


#%%

