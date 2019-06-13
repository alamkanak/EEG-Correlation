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
print(all_present)

#%% Function to open MEP file.
def open_mep_as_df(path):
    fileMep = open(path, "r+")
    mep_frames = fileMep.read().split('\n\n')
    df_list = []
    for mep_frame in mep_frames:
        df_list.append(pd.read_csv(StringIO(mep_frame), '\t'))
    return df_list

#%% Function to crop MEP region.
def crop_mep_region(mep_frame):
    crop_start = 0.211
    crop_end = 0.4
    multiplier = 7499 / 1.4998
    start = int(crop_start * multiplier)
    end = int(crop_end * multiplier)
    mep_cropped = mep_frame.iloc[start:end, :]
    time = mep_frame.iloc[start:end, :]
    return mep_cropped, time

#%% Function to get APB column name.
def get_apb_column_name(mep_frame):
    if 'L APB' in df:
        return 'L APB'
    else:
        return 'APB'

#%% Calculate the max and min peaks across all frames/epochs.
max_meps_among_all_frames_list = []
for mep in meps:
    mep_frames = open_mep_as_df(mep)
    max_in_frame_list = []
    for frame in range(len(mep_frames)):
        df = mep_frames[frame]
        mep_cropped, time = crop_mep_region(df)
        max_row = df.iloc[mep_cropped.idxmax(axis=0)[apb_name]]
        min_row = df.iloc[mep_cropped.idxmin(axis=0)[apb_name]]
        apb_name = get_apb_column_name(df)
        max_in_frame_list.append({
            'path': mep,
            'frame': frame,
            's-max': max_row['s'], 
            'apb-max': max_row[apb_name], 
            's-min': min_row['s'], 
            'apb-min': min_row[apb_name],
            'mep-size': max_row[apb_name] - min_row[apb_name]
            })
    df = pd.DataFrame(max_in_frame_list)
    df = df.sort_values(by=['mep-size'], ascending=False)
    max_meps_among_all_frames_list.append(df.iloc[0])
max_meps_among_all_frames_df = pd.DataFrame(max_meps_among_all_frames_list)
max_meps_among_all_frames_df = max_meps_among_all_frames_df.set_index('path')

#%% Identify peaks and plot them.
for mep in meps:
    #  Plot MEP
    df_list = open_mep_as_df(mep)
    for num in range(10):
        # Read a MEP frame.
        df = df_list[num]
        apb_name = get_apb_column_name(df)

        # Crop region of interest and delete stimulation artifact.
        mep_cropped, time = crop_mep_region(df)

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

        # Compare with max size.
        mep_size = max_row[apb_name] - min_row[apb_name]
        max_mep = max_meps_among_all_frames_df.loc[mep]
        # axs.annotate(str(mep_size) + '<=' + str(max_mep['mep-size']), 
        #             xy=(min_row.s + 0.23, min_row[apb_name]), 
        #             xytext=(min_row.s + 0.23, max_row[apb_name]),
        #             arrowprops=dict(facecolor='black', arrowstyle='|-|')
        #             )
        axs.axhline(min_row[apb_name], color='firebrick')
        axs.axhline(min_row[apb_name] + max_mep['mep-size'], color='firebrick')
        axs.axhline(min_row[apb_name] + max_mep['mep-size'] / 3, color='firebrick', linestyle='--')
        axs.axhline(min_row[apb_name] + max_mep['mep-size'] * 2 / 3, color='firebrick', linestyle='--')
        plt.show()
    break

#%%
# 