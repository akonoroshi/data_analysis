import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import preprocess

def plot_interactions(
    data: pd.DataFrame, X: list, y: str, label: str, name: str, max_plot_val=1.0, 
        boxplot=False, old_exp=[], non_bin=np.empty((1,1))):
    '''
    For each valid combination of independent (dummy) variables, plots interaction effects.
    Parameters:
        data (pandas.DataFrame): df containing data of the experiment.
        X (list): the list of the independent variables.
        y (str): the name of the dependent variable.
        label (str): the description of the dependent variable that goes to a y-axis of a figure.
        name (str): the version of the experiment.
        boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
        old_exp (list): the list of experiments that do not have any non-binary factors.
        non_bin (np.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each column in a raw represents a level.
    '''

    for c in itertools.combinations(X, 2): 
        if c[0] in np.delete(non_bin, 0, axis=1) or c[1] in np.delete(non_bin, 0, axis=1):
            # dummy variable of another factor
            continue
        fig, ax = plt.subplots(1, 2, figsize=(8,8))
        ax = ax.ravel()
        for i in range(2):
            mean_outcomes = []
            se_outcomes = []
            num_students = []
            no_yes = ['no', 'yes']
            if name not in old_exp and c[0] in non_bin[:,0]:
                # more than two levels
                base_data = data[data[c[1]] == i]
                factor = np.where(non_bin[:,0] == c[0])[0][0]
                groups = np.insert(non_bin[factor], 0, 'no')
                for level in non_bin[factor]:
                    base_data = base_data[base_data[level] == 0]
                if boxplot:
                    mean_outcomes.append(base_data[y])
                    for level in non_bin[factor]:
                        mean_outcomes.append(data[(data[level] == 1) & (
                            data[c[1]] == i)][y])
                else:
                    mean_outcomes.append(np.mean(base_data[y]))
                    se_outcomes.append(stats.sem(base_data[y]))
                    for level in non_bin[factor]:
                        mean_outcomes.append(np.mean(data[(data[level] == 1) & (
                            data[c[1]] == i)][y]))
                        se_outcomes.append(stats.sem(data[(data[level] == 1) & (
                            data[c[1]] == i)][y]))

                num_students.append(len(base_data))
                for level in non_bin[factor]:
                    num_students.append(len(data[(data[level] == 1) & (
                        data[c[1]] == i)])) 
            else:
                groups = no_yes
                if boxplot:
                    for j in range(2):
                        mean_outcomes.append(data[(data[c[0]] == j) & (
                            data[c[1]] == i)][y])
                else:
                    mean_outcomes = data[data[c[1]] == i].groupby(c[0], as_index=False)[y].mean()[y]
                    se_outcomes = data[data[c[1]] == i].groupby(c[0], as_index=False)[y].sem()[y]
                for j in range(2):
                    num_students.append(len(data[(data[c[1]] == i) & (data[c[0]] == j)]))

            x_pos = np.arange(len(groups))

            # Build the plot
            try:
                if boxplot:
                    ax[i].boxplot(mean_outcomes, labels=groups, whis=2.0)
                else:
                    ax[i].bar(x_pos, mean_outcomes, yerr=se_outcomes, align='center', 
                        alpha=0.5, ecolor='black', capsize=20)
                    ax[i].set_xticks(x_pos)
                    ax[i].set_xticklabels(groups, fontsize = 12)
            except ValueError as e:
                print('Not enough data for ' + c[0] + ' and ' + c[1])
                continue
            # Add text in graphs (mean and sample size)
            for j in range(len(groups)):
                if boxplot:
                    ax[i].text(j+1, 0.2*max_plot_val, 'Median =', ha='center', va='bottom',
                            fontweight='bold', fontsize = 16)
                    ax[i].text(j+1, 0.16*max_plot_val, str(mean_outcomes[
                        j].median()), ha='center', va='bottom', fontweight='bold', fontsize = 16)
                    ax[i].text(j+1, 0.05*max_plot_val, 'n = %s' %num_students[
                        j], ha='center', va='bottom', fontweight='bold', fontsize = 16)
                else:
                    ax[i].text(j, 0.2*max_plot_val, 'Mean =', ha='center', va='bottom', fontweight='bold', 
                        fontsize = 14)
                    ax[i].text(j, 0.16*max_plot_val, str(np.round(
                        mean_outcomes[j],2)), ha='center', va='bottom', fontweight='bold', 
                        fontsize = 14)
                    ax[i].text(j, 0.05*max_plot_val, 'n = %s' %num_students[
                        j], ha='center', va='bottom', fontweight='bold', fontsize = 14)

            ax[i].set_title(no_yes[i], fontsize = 16)
            ax[i].set_ylim(0, max_plot_val)
            ax[i].yaxis.grid(True)
   
        # Save the figure and show
        fig.suptitle(name.replace('_', ' ').title() + ': ' + c[1], fontsize = 18)
        fig.text(0.5, 0.04, c[0], ha='center', fontsize = 16)
        fig.text(0.04, 0.5, label, va='center', rotation='vertical', fontsize = 16)
        plt.show()
        
def plot_main(data: pd.DataFrame, X: list, y: str, label: str, name: str, 
    max_plot_val=1.0, boxplot=False, old_exp=[], non_bin=np.empty((1,1))):
    '''
    For each level of each factor, draws a boxplot if the dependent variable is continuous 
    or a horizonal bargraph if it is binary or categorical. 
    Parameters:
        data (pandas.DataFrame): df containing data of the experiment.
        X (list): the list of the independent variables.
        y (str): the name of the dependent variable.
        label: the description of the dependent variable that goes to a y-axis of a figure.
        name: the version of the experiment.
        top: the maximum y value (the value of the dependent variables) you show in figures.
        boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
        old_exp (list): the list of experiments that do not have any non-binary factors.
        non_bin (np.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each column in a raw represents a level.
    '''
    
    for x in X:
        # Create Arrays for the plot
        mean_outcomes = []
        se_outcomes = []
        num_students = []
        if x in np.delete(non_bin, 0, axis=1): 
            # dummy variable of another factor
            continue
        elif name not in old_exp and x in non_bin[:,0]:
            # more than two levels
            base_data = data
            factor = np.where(non_bin[:,0] == x)[0][0]
            groups = np.insert(non_bin[factor], 0, 'no')
            for level in non_bin[factor]:
                base_data = base_data[base_data[level] == 0]
            if boxplot: # a boxplot requires a 2D array
                mean_outcomes.append(base_data[y])
                for level in non_bin[factor]:
                    mean_outcomes.append(data[data[level] == 1][y])
            else:
                mean_outcomes.append(np.mean(base_data[y]))
                se_outcomes.append(stats.sem(base_data[y]))
                for level in non_bin[factor]:
                    mean_outcomes.append(np.mean(data[data[level] == 1][y]))
                    se_outcomes.append(stats.sem(data[data[level] == 1][y]))

            num_students.append(len(base_data))
            for level in non_bin[factor]:
                num_students.append(len(data[data[level] == 1]))         
        else: # factors with two levels
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
        try:
            fig, ax = plt.subplots()
            if boxplot:
                ax.boxplot(mean_outcomes, labels=groups, whis=2.0)
            else:
                ax.bar(x_pos, mean_outcomes, yerr=se_outcomes, align='center', 
                    alpha=0.5, ecolor='black', capsize=20)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(groups, fontsize = 16)
        except ValueError as e:
            print('Not enough data for ' + x)
            continue
        ax.set_ylabel(label, fontsize = 16)
        ax.set_xlabel(x, fontsize = 16)
        # Add text in praphs (mean and sample size)
        for i in range(len(groups)):
            if boxplot:
                ax.text(i+1, 0.2*max_plot_val, 'Median = %s' % np.round(
                    mean_outcomes[i].median(),2), ha='center', va='bottom',
                        fontweight='bold', fontsize = 16)
                ax.text(i+1, 0.05*max_plot_val, 'n = %s' %num_students[
                    i], ha='center', va='bottom', fontweight='bold', fontsize = 16)
            else:
                ax.text(i, 0.2*max_plot_val, 'Mean = %s' % np.round(
                    mean_outcomes[i],2), ha='center', va='bottom', fontweight='bold', 
                        fontsize = 16)
                ax.text(i, 0.05*max_plot_val, 'n = %s' %num_students[
                    i], ha='center', va='bottom', fontweight='bold', fontsize = 16)

        ax.set_title(name.replace('_', ' ').title(), fontsize = 18)
        ax.set_ylim(0, max_plot_val)
        ax.yaxis.grid(True)
   
        # Save the figure and show
        plt.show()

def explore(data: pd.DataFrame, data_dict: dict, X: list, y: str, ylabel: str, 
    max_plot_val=1.0, boxplot=False, old_exp=[], non_bin=np.empty((1,1))):
    '''
    Plots main effects and interaction effects for each experiment and the pooled result.
    Parameters:
        data (pandas.DataFrame): df containing overall data of the experiments
        data_dict (dict): the dictionary whose key is the names of experiments and 
        value is data (pandas.DataFrame). If you have only one experiment, you can set 
        this as an empty dict.
        X (list): the list of the independent variables.
        y (str): the name of the dependent variable.
        ylabel (str): the description of y that appears in graphs
        boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
        old_exp (list): the list of experiments that do not have any non-binary factors.
        non_bin (np.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each column in a raw represents a level.
    '''
    # For each experiment
    for name, prob in data_dict.items():
        relevant_X = preprocess.select_X(X, name)
        plot_main(prob, relevant_X, y, ylabel, name, max_plot_val, boxplot, old_exp, non_bin)
        plot_interactions(
            prob, relevant_X, y, ylabel, name, max_plot_val, boxplot, old_exp, non_bin)

    # Pooled result
    plot_main(data, X, y, ylabel, 'Overall', max_plot_val, boxplot, old_exp, non_bin)
    plot_interactions(
        data, X, y, ylabel, 'Overall', max_plot_val, boxplot, old_exp, non_bin)