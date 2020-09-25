import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.notebook import tqdm
from itertools import combinations
from scipy.stats import pearsonr
from matplotlib import colors as mcolors

def power_trial_plot(df_subject, band_name, def_values_power, hue_order_power, y_lim, show_legend):
    df_power2 = df_subject[df_subject['Band'] == band_name]
    with plt.style.context(['science-raquib2']):
        f, axs = plt.subplots(3, 2, figsize=(7,4))
        # f, axs = plt.subplots(2,3,figsize=(13, 8))
        i = 0
        j = 0
        for variable in tqdm(list(def_values_power.keys())):
            df_power3 = df_power2
            print('--------------')
            print(variable)
            print('--------------')
            for key, value in def_values_power.items():
                if key != variable:
                    df_power3 = df_power3[df_power3[key] == value]
                    print('Performing {} = {} for constant {}, shape = {}'.format(key, value, variable, df_power3.shape))

            df_power3 = df_power3[['trial_abs', 'Power', variable]]
            sns.lineplot(x="trial_abs", y="Power", hue=variable, data=df_power3, ax=axs[i, j], hue_order=hue_order_power[variable], palette=sns.color_palette("Set1", df_power3[variable].nunique()))
            
            
            
            if y_lim is not None:
                axs[i, j].set_ylim(y_lim)
            if show_legend == False:
                axs[i, j].get_legend().remove()
            axs[i,j].set_xlabel('Trial')
            if j == 0:
                axs[i, j].set_ylabel('Power (dB)')
            else:
                axs[i, j].set_ylabel('')
                if y_lim is not None:
                    axs[i, j].get_yaxis().set_ticks([])
            j = j + 1
            if j > 1:
                j = 0
                i = i + 1
        plt.tight_layout()

def power_box_plot(df_subject, def_values_power, hue_order_power, y_lim, y_ticks, show_legend):
    with plt.style.context(['science-raquib2']):
        f, axs = plt.subplots(1,5,figsize=(9, 2))
        i = 0
        for variable in tqdm(list(def_values_power.keys())):    
            df_power3 = df_subject
            print('--------------')
            print(variable)
            print('--------------')
            for key, value in def_values_power.items():
                if key != variable:
                    df_power3 = df_power3[df_power3[key] == value]
                    print('Performing {} = {} for constant {}, shape = {}'.format(key, value, variable, df_power3.shape))

            g=sns.boxplot(x="Band", y="Power", hue=variable, data=df_power3, ax=axs[i], fliersize=0, hue_order=hue_order_power[variable])
            if y_lim is not None:
                axs[i].set_ylim(y_lim)
            plt.setp(axs[i].lines, color='k')
            axs[i].set_title(variable)
            if show_legend == False:
                axs[i].get_legend().remove()
            axs[i].xaxis.set_ticks_position('bottom')
            if i == 0:
                axs[i].set_ylabel('Power (dB)')
                if y_ticks is not None:
                    axs[i].get_yaxis().set_ticks(y_ticks)
            else:
                axs[i].set_ylabel('')
                if y_ticks is not None:
                    axs[i].set_yticklabels([])
                    # axs[i].get_yaxis().set_ticks([])
            i = i + 1
            for _, box in enumerate(g.artists):
                box.set_edgecolor('black')
        plt.tight_layout()
        

def power_violin_plot(df_subject, def_values_power, hue_order_power, y_lim, y_ticks, show_legend):
    with plt.style.context(['science-raquib2']):
        f, axs = plt.subplots(1,5,figsize=(9, 2))
        i = 0
        for variable in tqdm(list(def_values_power.keys())):    
            df_power3 = df_subject
            print('--------------')
            print(variable)
            print('--------------')
            for key, value in def_values_power.items():
                if key != variable:
                    df_power3 = df_power3[df_power3[key] == value]
                    print('Performing {} = {} for constant {}, shape = {}'.format(key, value, variable, df_power3.shape))

            g=sns.violinplot(x="Band", y="Power", hue=variable, data=df_power3, ax=axs[i], fliersize=0, hue_order=hue_order_power[variable])
            if y_lim is not None:
                axs[i].set_ylim(y_lim)
            plt.setp(axs[i].lines, color='k')
            axs[i].set_title(variable)
            if show_legend == False:
                axs[i].get_legend().remove()
            axs[i].xaxis.set_ticks_position('bottom')
            if i == 0:
                axs[i].set_ylabel('Power (dB)')
                if y_ticks is not None:
                    axs[i].get_yaxis().set_ticks(y_ticks)
            else:
                axs[i].set_ylabel('')
                if y_ticks is not None:
                    axs[i].set_yticklabels([])
                    # axs[i].get_yaxis().set_ticks([])
            i = i + 1
            for _, box in enumerate(g.artists):
                box.set_edgecolor('black')
        plt.tight_layout()
        
def power_interaction_plot(df_subject, def_values_power, band_name, hue_order_power, y_lim, show_legend, hide_tick_labels):
    combs = list(combinations(list(def_values_power.keys()), 2))
    df_power2 = df_subject[df_subject['Band'] == band_name]
    i = 0
    j = 0
    with plt.style.context(['science-raquib2']):
        figsize = (7, 3)
        if hide_tick_labels == False:
            figsize = (12, 7)
        f, axs = plt.subplots(2, 5, figsize=figsize)
        for comb in tqdm(combs):
            df_power3 = df_power2.copy()
            for key, value in def_values_power.items():
                if key not in comb:
                    df_power3 = df_power3[df_power3[key] == value]
            print("Plotting in ({}, {}): {}, shape: {}, unique1: {}, unique2: {}".format(i, j, comb, df_power3.shape, df_power3[comb[0]].unique(), df_power3[comb[1]].unique()))
            df_power3 = df_power3[[comb[0], comb[1], 'Power']]
            sns.pointplot(x=comb[0], y="Power", hue=comb[1], data=df_power3, errwidth=0, ax=axs[i,j], order=hue_order_power[comb[0]], hue_order=hue_order_power[comb[1]])
            if show_legend == False:
                axs[i, j].get_legend().remove()
            if j == 0:
                axs[i, j].set_ylabel('Power (dB)')
            else:
                axs[i, j].set_ylabel('')
                axs[i, j].set_yticklabels([])
            axs[i, j].set_xlabel(comb[0])
            if hide_tick_labels == True:
                axs[i, j].set_xticklabels([])
                # axs[i, j].set_xlabel('')
            if y_lim is not None:
                axs[i, j].set_ylim(y_lim)
            j = j + 1
            if j > 4:
                i = i + 1
                j = 0
        plt.tight_layout()
        
def phase_hist_plot(df_subject, def_values_phase, y_lim):
    with plt.style.context(['science-raquib2']):
        colors = ['#EF5350', '#43A047', '#039BE5']
        f, axs = plt.subplots(3, 4, figsize=(7,5))
        i = 0
        j = 0
        color_ind = 0
        for variable in tqdm(list(def_values_phase.keys())):
            df_phase3 = df_subject.copy()
            for key, value in def_values_phase.items():
                    if key != variable:
                        df_phase3 = df_phase3[df_phase3[key] == value]
            for band in ['Theta', 'Mu', 'Beta', 'Gamma']:
                print('{}, Band: {}, Options: {}'.format(variable, band, df_phase3[variable].unique()))
                for option in df_phase3[variable].unique():
                    df_phase4 = df_phase3[df_phase3['Band'] == band]
                    df_phase4 = df_phase4[df_phase4[variable] == option]
                    sns.distplot(df_phase4['Phase'], hist=False, color=colors[color_ind], kde_kws={"shade": True}, ax=axs[i, j])
                    color_ind = color_ind + 1
                    if i == 2:
                        axs[i, j].set_xlabel('Phase (degrees)')
                    else:
                        axs[i, j].set_xlabel('')
                    axs[i, j].set_xlim([0, 360])
                    axs[i, j].set_ylim(y_lim)
                    axs[i, j].set_xticks([0, 90, 180, 270, 360])
                if i == 0:
                    axs[i, j].set_title(band)
                if j == 0:
                    axs[i, j].set_ylabel('Density')
                else:
                    axs[i, j].get_yaxis().set_visible(False)
                j = j + 1
                if j > 3:
                    j = 0
                color_ind = 0
            i = i + 1

        plt.tight_layout()
        
def power_corr_plot(df_corr, x_lim, x_ticks):
    with plt.style.context(['science-raquib2']):
        colors = ['#f44336', '#7e57c2', '#29b6f6', '#ffeb3b']
        fig, ax = plt.subplots(figsize=(1.5,2.8))
        g = sns.boxplot(y='factor_value', x='Correlation', hue='band', data=df_corr, orient='h', ax=ax, fliersize=0.05, linewidth=0.3, palette=[mcolors.to_rgba(c) for c in colors])
        ax.tick_params(axis='y', which='both', length=0)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.get_legend().remove()
        plt.setp(ax.lines, color='k')
        for pos in ['top','bottom','left','right']:
            ax.spines[pos].set_linewidth(0.3)
        for i, box in enumerate(g.artists):
            box.set_edgecolor('black')
        ax.xaxis.set_ticks_position('bottom')
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            
def power_corr_plot_violin(df_corr, x_lim, x_ticks):
    with plt.style.context(['science-raquib2']):
        colors = ['#f44336', '#7e57c2', '#29b6f6', '#ffeb3b']
        fig, ax = plt.subplots(figsize=(1.5,2.8))
        g = sns.violinplot(y='factor_value', x='Correlation', hue='band', data=df_corr, orient='h', ax=ax, fliersize=0.05, linewidth=0.3, palette=[mcolors.to_rgba(c) for c in colors])
        ax.tick_params(axis='y', which='both', length=0)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.get_legend().remove()
        plt.setp(ax.lines, color='k')
        for pos in ['top','bottom','left','right']:
            ax.spines[pos].set_linewidth(0.3)
        for i, box in enumerate(g.artists):
            box.set_edgecolor('black')
        ax.xaxis.set_ticks_position('bottom')
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        
def phase_corr_plots(df_corr, x_lim, x_ticks):
    with plt.style.context(['science-raquib2']):
        for grp in ['peak', 'trough']:
            df_corr2 = df_corr[df_corr['group'] == grp]
            colors = ['#f44336', '#7e57c2', '#29b6f6', '#ffeb3b']
            fig, ax = plt.subplots(figsize=(1.5,2))
            g = sns.boxplot(y='factor_value', x='Correlation', hue='band', data=df_corr2, orient='h', ax=ax, fliersize=0.05, linewidth=0.3, palette=[mcolors.to_rgba(c) for c in colors])
            ax.tick_params(axis='y', which='both', length=0)
            if x_lim is not None:
                ax.set_xlim(x_lim)
            ax.get_legend().remove()
            plt.setp(ax.lines, color='k')
            for pos in ['top','bottom','left','right']:
                ax.spines[pos].set_linewidth(0.3)
            for i, box in enumerate(g.artists):
                box.set_edgecolor('black')
            ax.xaxis.set_ticks_position('bottom')
            if x_ticks is not None:
                ax.set_xticks(x_ticks)