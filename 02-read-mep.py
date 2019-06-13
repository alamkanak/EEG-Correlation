#%%
import glob
import pandas as pd
import mne
import matplotlib.pyplot as plt
from io import StringIO

#%%
experiments = sorted(glob.glob('data/original/*/*'))
i = 0
exclude_subs = [
    ['sub01', '*', '*']
]

for experiment in experiments:
    # Ignore excluded subjects and experiments (e.g. MND or bad data).
    print(experiment)
    subject = experiment.split('/')[2]
    exp = experiment.split('/')[3]

    exclude_exps = [row for row in exclude_subs 
        if (subject in row[0] and exp in row[1]) or (subject in row[0] and '*' in row[1])
    ]
    skip = False
    for exp in exclude_exps:
        if (exp[1] == '*' or exp[2] == '*'):
            print('  Skipping experiment')
            skip = True
            break
    if skip:
        continue

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

            # Skip this RMT if it should be ignored.
            skip = False
            for exp in exclude_exps:
                if exp[2] == mep.split('/')[5]:
                    skip = True
                    print('  Skipping ' + mep.split('/')[5])
                    break
            if skip:
                continue
            print('  Doing ' + mep.split('/')[5])

            #  Plot MEP
            fileMep = open(mep, "r+")
            mepFrames = fileMep.read().split('\n\n')
            for num in range(1):
                # Read a MEP frame.
                df = pd.read_csv(StringIO(mepFrames[num]), '\t')

                # Delete unused channel.

                # Plot the MEP.
                plt.figure(figsize=(14, 5))
                if 'L APB' in df:
                    plt.subplot(2, 1, 1)
                    plt.plot(df['s'], df['L APB'])
                    plt.title('L APB')
                else:
                    plt.subplot(2, 1, 1)
                    plt.plot(df['s'], df['APB'])
                    plt.title('APB')
                plt.suptitle(mep)
                plt.show()
        #     break
        # break


#%%