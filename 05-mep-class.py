#%%
import glob
import pandas as pd
import mne
import matplotlib.pyplot as plt
from io import StringIO

#%%
experiment = 'data/original/sub06/exp01'

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
        for num in range(10):
            # Read a MEP frame.
            df = pd.read_csv(StringIO(mepFrames[num]), '\t')
            if 'L APB' in df:
                apb_name = 'L APB'
            else:
                apb_name = 'APB'

            # Crop region of interest and delete stimulation artifact.
            crop_start = 0.211
            crop_end = 0.4
            multiplier = 7499 / 1.4998
            start = int(crop_start * multiplier)
            end = int(crop_end * multiplier)
            mep_cropped = df.iloc[start:end, :]
            time = df.iloc[start:end, :]

            # Calculate peaks.
            max_row = df.iloc[mep_cropped.idxmax(axis=0)[apb_name]]
            min_row = df.iloc[mep_cropped.idxmin(axis=0)[apb_name]]

            # Plot the MEP.
            plt.figure(figsize=(14, 5))
            fig, axs = plt.subplots(1, 1, figsize=(14, 5))
            axs.plot(df['s'], df[apb_name])
            axs.set_title(apb_name + ' Frame ' + str(num))
            plt.axvspan(crop_start, crop_end, color='red', alpha=0.1)
            plt.suptitle(mep)
            
            # Indicate peaks.
            axs.plot(max_row.s, max_row[apb_name], 'ro')
            axs.plot(min_row.s, min_row[apb_name], 'ro')
            axs.annotate(str(max_row[apb_name]), (max_row.s, max_row[apb_name]))
            axs.annotate(str(min_row[apb_name]), (min_row.s, min_row[apb_name]))
            plt.show()
        break

#%%