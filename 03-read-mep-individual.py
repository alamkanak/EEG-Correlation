#%%
import glob
import pandas as pd
import mne
import matplotlib.pyplot as plt
from io import StringIO

#%%
experiment = 'data/original/sub12/exp01*'

# Check if CMAP, MEP and EEG are all available.
cmaps = sorted(glob.glob(experiment + '/cmap/*.xlsx'))
cmap_present = len(cmaps) > 0
meps = sorted(glob.glob(experiment + '/mep/*/*.txt'))
mep_present = len(meps) > 0
eegs = sorted(glob.glob(experiment + '/eeg/*/*.cnt'))
eeg_present = len(eegs) > 0
all_present = cmap_present and mep_present and eeg_present
if all_present:
    for mep in meps:
        #  Plot MEP
        fileMep = open(mep, "r+")
        mepFrames = fileMep.read().split('\n\n')
        for num in range(len(mepFrames)):
            # Read a MEP frame.
            df = pd.read_csv(StringIO(mepFrames[num]), '\t')

            # Plot the MEP.
            plt.figure(figsize=(14, 5))
            if 'L APB' in df:
                plt.subplot(2, 1, 1)
                plt.plot(df['s'], df['L APB'])
                plt.title('L APB Frame ' + str(num))
            else:
                plt.subplot(2, 1, 1)
                plt.plot(df['s'], df['APB'])
                plt.title('APB Frame ' + str(num))
            plt.suptitle(mep)
            plt.show()

#%%