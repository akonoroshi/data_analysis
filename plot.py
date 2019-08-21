import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import preprocess

def plot_interactions(
    data: pd.DataFrame, X: list, y: str, label: str, name: str, max_plot_val=1.0):
    '''
    For each valid combination of independent (dummy) variables, plots interaction effects.
    Parameters:
        label: the description of the dependent variable that goes to a title of a figure.
        name: the version of the experiment.
    '''

    for c in itertools.combinations(X, 2): 
        if c[0] == 'research' or c[1] == 'research' or set(c) == {'instructor', 'how'}:
            # research is a dummy variable of factor motivate
            # instructor was removed and how was added in week 12 perform
            continue
        fig, ax = plt.subplots(1, 2, figsize=(8,8))
        ax = ax.ravel()
        for i in range(2):
            mean_outcomes = []
            se_outcomes = []
            num_students = []
            no_yes = ['no', 'yes']
            if name not in ['W10_insert', 'W10_bubble', 'W11_prepare'] and c[0] == 'motivate':
                # motivate has three levels
                groups = ['no', 'stop', 'research']
                mean_outcomes.append(np.mean(data[(data['motivate'] == 0) & (
                    data['research'] == 0) & (data[c[1]] == i)][y]))
                mean_outcomes.append(np.mean(data[(data['motivate'] == 1) & (
                    data[c[1]] == i)][y]))
                mean_outcomes.append(np.mean(data[(data['research'] == 1) & (
                    data[c[1]] == i)][y]))

                se_outcomes.append(stats.sem(data[(data['motivate'] == 0) & (
                    data['research'] == 0) & (data[c[1]] == i)][y]))
                se_outcomes.append(stats.sem(data[(data['motivate'] == 1) & (
                    data[c[1]] == i)][y]))
                se_outcomes.append(stats.sem(data[(data['research'] == 1) & (
                    data[c[1]] == i)][y]))

                num_students.append(len(data[(data['motivate'] == 0) & (
                    data['research'] == 0) & (data[c[1]] == i)][y]))
                num_students.append(len(data[(data['motivate'] == 1) & (
                    data[c[1]] == i)][y]))
                num_students.append(len(data[(data['research'] == 1) & (
                    data[c[1]] == i)][y]))
            else:
                groups = no_yes
                mean_outcomes = data[data[c[1]] == i].groupby(c[0], as_index=False)[y].mean()[y]
                se_outcomes = data[data[c[1]] == i].groupby(c[0], as_index=False)[y].sem()[y]
                for j in range(2):
                    num_students.append(len(data[(data[c[1]] == i) & (data[c[0]] == j)]))

            x_pos = np.arange(len(groups))

            # Build the plot
            ax[i].bar(x_pos, mean_outcomes, yerr=se_outcomes, align='center', 
                alpha=0.5, ecolor='black', capsize=20)
            ax[i].set_xticks(x_pos)
            ax[i].set_xticklabels(groups, fontsize = 12)
            # Add text in histograms (mean and sample size)
            for j in range(len(groups)):
                ax[i].text(j, 0.2*max_plot_val, 'Mean =', ha='center', va='bottom', fontweight='bold', 
                    fontsize = 14)
                ax[i].text(j, 0.16*max_plot_val, str(np.round(
                    mean_outcomes[j],2)), ha='center', va='bottom', fontweight='bold', 
                    fontsize = 14)
                ax[i].text(j, 0.05*max_plot_val, 'n = %s' %num_students[
                    j], ha='center', va='bottom', fontweight='bold', fontsize = 14)

            ax[i].set_title(no_yes[i], fontsize = 16)
            ax[i].set_ylim(top=max_plot_val)
            ax[i].yaxis.grid(True)
   
        # Save the figure and show
        fig.suptitle(name.replace('_', ' ').title() + ': ' + c[1], fontsize = 18)
        fig.text(0.5, 0.04, c[0], ha='center', fontsize = 16)
        fig.text(0.04, 0.5, label, va='center', rotation='vertical', fontsize = 16)
        plt.show()
        
def plot_main(data: pd.DataFrame, X: list, y: str, label: str, name: str, 
    max_plot_val=1.0, boxplot=False):
    '''
    For each level of each factor, draws a boxplot if the dependent variable is continuous 
    or a horizonal bargraph if it is binary or categorical. 
    Parameters:
        label: the description of the dependent variable that goes to a title of a figure.
        name: the version of the experiment.
        top: the maximum y value (the value of the dependent variables) you show in figures.
    '''
    
    for x in X:
        # Create Arrays for the plot
        mean_outcomes = []
        se_outcomes = []
        num_students = []
        if x == 'research': # research is a dummy variable of factor motivate
            continue
        elif name not in ['W10_insert', 'W10_bubble', 'W11_prepare'] and x == 'motivate':
            # motivate has three levels
            if boxplot: # a boxplot requires a 2D array
                mean_outcomes.append(data[(data['motivate'] == 0) & (data['research'] == 0)][y])
                mean_outcomes.append(data[data['motivate'] == 1][y])
                mean_outcomes.append(data[data['research'] == 1][y])
            else:
                mean_outcomes.append(np.mean(data[(data['motivate'] == 0) & (
                    data['research'] == 0)][y]))
                mean_outcomes.append(np.mean(data[data['motivate'] == 1][y]))
                mean_outcomes.append(np.mean(data[data['research'] == 1][y]))

                se_outcomes.append(stats.sem(data[(data['motivate'] == 0) & (
                    data['research'] == 0)][y]))
                se_outcomes.append(stats.sem(data[data['motivate'] == 1][y]))
                se_outcomes.append(stats.sem(data[data['research'] == 1][y]))

            groups = ['no', 'stop', 'research']

            num_students.append(len(data[(data['motivate'] == 0) & (
                data['research'] == 0)]))
            num_students.append(len(data[data['motivate'] == 1]))
            num_students.append(len(data[data['research'] == 1]))            
        else: # other factors have only two levels
            if boxplot:
                for i in range(2):
                    mean_outcomes.append(data[data[x] == i][y])
            else:
                mean_outcomes = data.groupby(x, as_index=False)[y].mean()[y]
                se_outcomes = data.groupby(x, as_index=False)[y].sem()[y]
            groups = ['no', 'yes']
            for i in range(2):
                num_students.append(len(data[data[x] == i]))

        x_pos = np.arange(len(groups))

        # Build the plot
        fig, ax = plt.subplots()
        if boxplot:
            ax.boxplot(mean_outcomes, labels=groups, whis=2.0)
        else:
            ax.bar(x_pos, mean_outcomes, yerr=se_outcomes, align='center', 
                alpha=0.5, ecolor='black', capsize=20)
        ax.set_ylabel(label, fontsize = 16)
        ax.set_xlabel(x, fontsize = 16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, fontsize = 16)
        # Add text in histograms (mean and sample size)
        for i in range(len(groups)):
            ax.text(i, 0.2*max_plot_val, 'Mean = %s' % np.round(
                mean_outcomes[i],2), ha='center', va='bottom', fontweight='bold', 
                    fontsize = 16)
            ax.text(i, 0.05*max_plot_val, 'n = %s' %num_students[
                i], ha='center', va='bottom', fontweight='bold', fontsize = 16)

        ax.set_title(name.replace('_', ' ').title(), fontsize = 18)
        ax.set_ylim(top=max_plot_val)
        ax.yaxis.grid(True)
   
        # Save the figure and show
        plt.show()

def explore(data: pd.DataFrame, data_dict: dict, X: list, y: str, 
    ylabel: str, max_plot_val=1.0, boxplot=False):
    '''
    Plots main effects and interaction effects for each experiment and the pooled result.
    '''
    # For each experiment
    for name, prob in data_dict.items():
        relevant_X = preprocess.select_X(X, name)
        plot_main(prob, relevant_X, y, ylabel, name, max_plot_val, boxplot)
        plot_interactions(prob, relevant_X, y, ylabel, name, max_plot_val)

    # Pooled result
    plot_main(data, X, y, ylabel, 'Overall', max_plot_val, boxplot)
    plot_interactions(data, X, y, ylabel, 'Overall', max_plot_val)