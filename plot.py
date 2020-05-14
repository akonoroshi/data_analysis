import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import preprocess

####################################################################################################
# The functions located in the upper half of this file are all helper functions and you should not #
# call them directly. Scroll down to find the functions you can use as a blackbox.                 #
####################################################################################################

def build_plot(ax, data: pd.DataFrame, factor: str, factor_levels: np.ndarray, y: str,
    name: str, title: str, max_plot_val=1.0, boxplot=False, min_plot_val=0.0):
    '''
    Builds a plot of the effect of the factor on y
    Parameters
    ----------
    ax: an axes object in matplotlib
    data (pandas.DataFrame): df containing data of the experiment.
    factor (str): the name of the independent variable.
    factor_levels (numpy.ndarray): the list of levels of factor.
    y (str): the name of the dependent variable.
    name (str): the version of the experiment.
    title (str): the title of this axes.
    max_plot_val (float): the maximum value of the plot. 
    min_plot_val (float): the minimum value of the plot. 
    boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
    '''
    # Create Arrays for the plot
    mean_outcomes = []
    se_outcomes = []
    num_students = []
    groups = factor_levels
    base_data = data # reference level
    for i in range(1, len(factor_levels)):
        base_data = base_data[base_data[factor + '_' + factor_levels[i]] == 0]
        assigned = data[
            data[factor + '_' + factor_levels[i]] == 1]
        num_students.append(len(assigned))
        if boxplot: # a boxplot requires a 2D array
            mean_outcomes.append(assigned[y])
        else:
            mean_outcomes.append(np.mean(assigned[y]))
            se_outcomes.append(stats.sem(assigned[y]))

    num_students.insert(0, len(base_data))
    if boxplot: # a boxplot requires a 2D array
        mean_outcomes.insert(0, base_data[y])
    else:
        mean_outcomes.insert(0, np.mean(base_data[y]))
        se_outcomes.insert(0, stats.sem(base_data[y]))
            
    x_pos = np.arange(len(groups))

    # Build the plot
    if boxplot:
        ax.boxplot(mean_outcomes, labels=groups, whis=2.0)
    else:
        ax.bar(x_pos, mean_outcomes, yerr=se_outcomes, align='center', 
            alpha=0.5, ecolor='black', capsize=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, fontsize = 12)
    # Add text in graphs (mean and sample size)
    for i in range(len(groups)):
        if boxplot:
            ax.text(i+1, 0.2*(max_plot_val - min_plot_val) + min_plot_val, 'Median =', ha='center', va='bottom',
                    fontweight='bold', fontsize = 16)
            ax.text(i+1, 0.15*(max_plot_val - min_plot_val) + min_plot_val, str(np.round(mean_outcomes[
                i].median(), 2)), ha='center', va='bottom', fontweight='bold', fontsize = 16)
            ax.text(i+1, 0.05*(max_plot_val - min_plot_val) + min_plot_val, 'n = %s' %num_students[
                i], ha='center', va='bottom', fontweight='bold', fontsize = 16)
        else:
            ax.text(i, 0.2*(max_plot_val - min_plot_val) + min_plot_val, 'Mean =', ha='center', va='bottom', fontweight='bold', 
                fontsize = 14)
            ax.text(i, 0.15*(max_plot_val - min_plot_val) + min_plot_val, str(np.round(
                mean_outcomes[i],2)), ha='center', va='bottom', fontweight='bold', 
                fontsize = 14)
            ax.text(i, 0.05*(max_plot_val - min_plot_val) + min_plot_val, 'n = %s' %num_students[
                i], ha='center', va='bottom', fontweight='bold', fontsize = 14)

    ax.set_title(title, fontsize = 16)
    ax.set_ylim(min_plot_val, max_plot_val)
    ax.yaxis.grid(True)

def plot_interactions_helper(data: pd.DataFrame, factor1: str, factor1_levels: np.ndarray, 
    factor2: str, factor2_levels: np.ndarray, y: str, ylabel: str, name: str, 
        max_plot_val=1.0, boxplot=False, subtitle='', min_plot_val=0.0):
    '''
    Plots interaction effects between factor1 and factor2 on y (dependent variable).
    Parameters
    ----------
    data (pandas.DataFrame): df containing data of the experiment.
    factor1 (str): the name of the first independent variable. Should be an action variable.
    factor1_levels (numpy.ndarray): the list of levels of factor1.
    factor2 (str): the name of the second independent variable. Can be either an action 
    variable or a contextual variable.
    factor2_levels (numpy.ndarray): the list of levels of factor2.
    y (str): the name of the dependent variable.
    ylabel (str): the description of the dependent variable that goes to a y-axis of a figure.
    name (str): the version of the experiment.
    max_plot_val (float): the maximum value of the plot. 
    min_plot_val (float): the minimum value of the plot. 
    boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
    '''

    fig, ax = plt.subplots(1, len(factor2_levels), figsize=(4*len(factor2_levels), 6))
    ax = ax.ravel()
    base_data = data # reference level

    for i in range(1, len(factor2_levels)):
        base_data = base_data[base_data[factor2 + '_' + factor2_levels[i]] == 0]
        filtered = data[data[factor2 + '_' + factor2_levels[i]] == 1]
        try:
            build_plot(ax[i], filtered, factor1, factor1_levels, y, name, 
                factor2_levels[i], max_plot_val, boxplot, min_plot_val)
        except (ValueError, KeyError) as e:
            print('Not enough data for ' + factor1 + ' and ' + factor2)
            continue
    try:
        build_plot(ax[0], base_data, factor1, factor1_levels, y, name, 
            factor2_levels[0], max_plot_val, boxplot, min_plot_val)
    except (ValueError, KeyError) as e:
        print('Not enough data for ' + factor1 + ' and ' + factor2)
    finally:   
        # Save the figure and show
        fig.suptitle(name.replace('_', ' ').title() + ': ' + factor2, fontsize = 18)
        fig.text(0.5, 0.04, factor1, ha='center', fontsize = 16)
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize = 16)
        fig.text(0.5, 0.9, subtitle, ha='center', fontsize = 16)
        plt.show()

def dummy_plot(title: str):
    '''
    Generates dummy plots to separate other plots.
    Parameters
    ----------
    title (str): the title of the dummpy plot.
    '''
    fig = plt.figure(figsize=(8, 1))
    plt.axes([0, 0, 0.1, 0.1])
    fig.suptitle(title, fontsize=20)
    plt.show()

####################################################################################################
# The functions above this are all helper functions and you should not call them directly.         #
# You can use the functions below as a blackbox.                                                   #
####################################################################################################

def plot_interactions(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, 
    y: str, ylabel: str, name: str, contexts=[], context_levels=[[]], 
        max_plot_val=1.0, boxplot=False, subtitle='', min_plot_val=0.0):
    '''
    For each valid combination of independent (dummy) variables, plots interaction effects.
    Parameters
    ----------
    data (pandas.DataFrame): df containing data of the experiment.
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each row represents a factor 
    and each element in a row represents a level.
    y (str): the name of the dependent variable.
    ylabel (str): the description of the dependent variable that goes to a y-axis of a figure.
    name (str): the version of the experiment.
    contexts (numpy.ndarray): the list of contextual variables. Set to be None if you 
    want to plot the interactions between action variables.
    context_levels (numpy.ndarray): the matrix of context x level. Each row represents a context 
    and each element in a row represents a level.
    max_plot_val (float): the maximum value of the plot. 
    min_plot_val (float): the minimum value of the plot. 
    boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
    '''

    if len(contexts) != 0:
        for factor, factor_levels in zip(factors, levels):
            for context, context_level in zip(contexts, context_levels):
                plot_interactions_helper(data, factor, factor_levels, context, context_level, 
                    y, ylabel, name, max_plot_val, boxplot, subtitle, min_plot_val)

    else:
        for c in itertools.combinations(range(len(factors)), 2):
            plot_interactions_helper(data, factors[c[0]], levels[c[0]], factors[c[1]], levels[c[1]], 
                y, ylabel, name, max_plot_val, boxplot, subtitle, min_plot_val)
        
def plot_main(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, y: str, 
    ylabel: str, name: str, max_plot_val=1.0, boxplot=False, min_plot_val=0.0):
    '''
    Plots main effects for each factor.
    Parameters
    ----------
    data (pandas.DataFrame): df containing data of the experiment.
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each raw represents a factor 
    and each element in a raw represents a level.
    y (str): the name of the dependent variable.
    ylabel (str): the description of the dependent variable that goes to a y-axis of a figure.
    name (str): the version of the experiment.
    max_plot_val (float): the maximum value of the plot. 
    min_plot_val (float): the minimum value of the plot. 
    boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
    '''
    
    for factor, factor_levels in zip(factors, levels):
        fig, ax = plt.subplots()
        try:
            build_plot(ax, data, factor, factor_levels, y, name, name.replace(
                '_', ' ').title(), max_plot_val, boxplot, min_plot_val)
        except ValueError as e:
            print('Not enough data for ' + factor)
            continue
        ax.set_ylabel(ylabel, fontsize = 16)
        ax.set_xlabel(factor, fontsize = 16)
   
        # Save the figure and show
        plt.show()

def plot_main_drop(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, y: str, 
    ylabel: str, name: str, can_be_dropped: list, max_plot_val=1.0, boxplot=False, min_plot_val=0.0):
    '''
    Plots main effects for each factor while dropping some of the data according to criteria
    specified by can_be_dropped.
    Parameters
    ----------
    data (pandas.DataFrame): df containing overall data of the experiments
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each raw represents a factor 
    and each element in a raw represents a level.
    y (str): the name of the dependent variable.
    ylabel (str): the description of the dependent variable that goes to a y-axis of a figure.
    name (str): the version of the experiment.
    can_be_dropped (list): the list of different critaria to clean data. Set column value
    to be 1 if you want to include the row. Set this to be an empty list if you don't have
    any other critaria to clean data.
    max_plot_val (float): the maximum value of the plot. 
    min_plot_val (float): the minimum value of the plot. 
    boxplots (list): the list of booleans. Must be the same length as ys. True if you 
    want to use boxplots. False if you want bar graphs.
    '''

    for factor, factor_levels in zip(factors, levels):
        fig, ax = plt.subplots(1, len(can_be_dropped) + 1, figsize=(20,6))
        ax = ax.ravel()
        try:
            build_plot(ax[0], data, factor, factor_levels, y, name, 'Nothing dropped', max_plot_val, boxplot, min_plot_val)
            for i in range(len(can_be_dropped)):
                temp = data[data[can_be_dropped[i]] == 1]
                build_plot(ax[i+1], temp, factor, factor_levels, y, name, 
                    'Dropped:' + can_be_dropped[i], max_plot_val, boxplot, min_plot_val)
        except ValueError as e:
            print('Not enough data for ' + factor)
        finally:
            fig.suptitle(name.replace('_', ' ').title(), fontsize = 18)
            fig.text(0.5, 0.04, factor, ha='center', fontsize = 16)
            fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize = 16)
            plt.show()

def discrete_hist(data, min_val: int, max_val: int, step: int, title: str, density=False):
    '''
    Builds a histogram of the discrete data.
    Parameters
    ----------
    data (array-like): data you want to plot on a histogram.
    min_val (int): the minimum value in the data (inclusive).
    max_val (int): the maximum value in the data (exclusive, add 1).
    step (int): the incrementation
    title (str): the title of this axes.
    density (bool): True if you want a density plot. False if you want counts.
    '''
    plt.hist(data, bins=np.arange(min_val - step/2, max_val + step/2, step), alpha=0.5, density=density)
    plt.title(title)
    plt.xticks(range(min_val, max_val, step))
    plt.show()

def assignment_percent(data: pd.DataFrame, factor: str, factor_levels: list, partition: int, reward: str, 
                    xlabel: str, sort_by: str, ascending: bool):
    '''
    Builds a plot of the assignments to different experimental versions over time.
    Parameters
    ----------
    data (pandas.DataFrame): df containing data of the experiment.
    factor (str): the name of the independent variable.
    factor_levels (numpy.ndarray): the list of levels of factor.
    partition (int): the number of partitions you want to divide data into.
    reward (str): the name of the dependent variable.
    xlabel (str): the name of the experiment.
    sort_by (str): the name of column that you want to sort data by. 
    ascending (bool): True if you want to sort in the ascending order. False for descending.
    '''

    data = data.sort_values(by=sort_by, ascending=ascending)    
    size_list = []
    for i in range(partition):
        if i <= data.shape[0] % partition:
            size_list.append(data.shape[0] // partition + 1)
        else:
            size_list.append(data.shape[0] // partition)
            
    cur_index = 0
    percent = np.zeros((len(factor_levels), partition))
    means = np.zeros((len(factor_levels), partition))
    for i in range(len(size_list)):
        part = data.iloc[cur_index:cur_index+size_list[i]]
        base_part = part.copy()
        base_level = 100
        for j in range(1, len(factor_levels)):
            normalized = part[factor + '_' + factor_levels[j]].sum() / size_list[i] * 100
            percent[j, i] += normalized
            means[j, i] += np.round(part[part[factor + '_' + factor_levels[j]] == 1][reward].mean(), 2)
            base_part = base_part[base_part[factor + '_' + factor_levels[j]] == 0]
            base_level -= normalized
        percent[0, i] += base_level
        means[0, i] += np.round(base_part[reward].mean(), 2)
        cur_index += size_list[i]
        
    bar_width = 1
    edgecolor = 'white'
    fig, ax = plt.subplots(figsize=((partition / 4 + 1) * 2, partition / 4 + 1))
    r = np.arange(partition)
    for i in range(len(factor_levels)):
        if i == 0:
            ax.bar(r, percent[i], edgecolor=edgecolor, width=bar_width, label=factor_levels[i])
            for j in range(partition):
                ax.annotate(means[i, j], (r[j], percent[i, j] / 2), ha='center')
        else:
            ax.bar(r, percent[i], edgecolor=edgecolor, width=bar_width, label=factor_levels[i], 
                    bottom=np.sum(percent[:i], axis=0))
            for j in range(partition):
                ax.annotate(means[i, j], (r[j], percent[i, j] / 2 + np.sum(percent[:i, j])), ha='center')
            
    ax.set_xlabel(xlabel)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    ax.set_title(factor)
    plt.show()

def explore(data: pd.DataFrame, data_dict: dict, factors: np.ndarray, levels: np.ndarray, y: str, ylabel: str, 
    max_plot_val=1.0, boxplot=False, contexts=None, context_levels=None, min_plot_val=0.0):
    '''
    Plots main effects and interaction effects for each experiment and the pooled result.
    Parameters
    ----------
    data (pandas.DataFrame): df containing overall data of the experiments
    data_dict (dict): the dictionary whose key is the names of experiments and 
    value is data (pandas.DataFrame). If you have only one experiment, you can set 
    this as an empty dict.
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each raw represents a factor 
    and each element in a raw represents a level.
    y (str): the name of the dependent variable.
    ylabel (str): the description of y that appears in graphs.
    max_plot_val (float): the maximum value of the plot. 
    min_plot_val (float): the minimum value of the plot. 
    boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
    contexts (numpy.ndarray): the list of contextual variables. Set to be None if you 
    want to plot the interactions between action variables.
    context_levels (numpy.ndarray): the matrix of context x level. Each row represents a context 
    and each element in a row represents a level.
    '''
    # For each experiment
    for name, prob in data_dict.items():
        indices = preprocess.select_factors(factors, name)
        relevant_factors = factors[indices]
        relevant_levels = levels[indices]
        plot_main(prob, relevant_factors, relevant_levels, y, ylabel, 
            name, max_plot_val, boxplot, min_plot_val)
        plot_interactions(prob, relevant_factors, relevant_levels, y, ylabel, 
            name, contexts, context_levels, max_plot_val, boxplot, min_plot_val=min_plot_val)

    # Pooled result
    plot_main(data, factors, levels, y, ylabel, 'Overall', max_plot_val, boxplot, min_plot_val)
    plot_interactions(data, factors, levels, y, ylabel, 'Overall', contexts, context_levels, max_plot_val, boxplot, min_plot_val=min_plot_val)

def explore_by_factor(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, ys: list, ylabels: list, name: str, 
    can_be_dropped: list, max_plot_vals: list, min_plot_vals: list, boxplots: list, contexts=None, context_levels=None):
    '''
    Plots main effects and interaction effects for each experiment and the pooled result 
    by factors.
    Parameters
    ----------
    data (pandas.DataFrame): df containing overall data of the experiments
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each raw represents a factor 
    and each element in a raw represents a level.
    ys (list): the list of names of the dependent variables.
    ylabels (list): the descriptions of y's that appear in graphs. Must be the same 
    length as ys.
    can_be_dropped (list): the list of different critaria to clean data. Set column value
    to be 1 if you want to include the row. Set this to be an empty list if you don't have
    any other critaria to clean data.
    max_plot_vals (list): the list of maximum values in each plot. Must be the same 
    length as ys.
    min_plot_vals (list): the list of minimum values in each plot. Must be the same 
    length as ys.
    boxplots (list): the list of booleans. Must be the same length as ys. True if you 
    want to use boxplots. False if you want bar graphs.
    contexts (numpy.ndarray): the list of contextual variables. Set to be None if you 
    want to plot the interactions between action variables.
    context_levels (numpy.ndarray): the matrix of context x level. Each row represents a context 
    and each element in a row represents a level.
    '''

    indices = preprocess.select_factors(factors, name)
    relevant_factors = factors[indices]
    relevant_levels = levels[indices]

    # Main effects
    print('Main effects')
    for factor, factor_levels in zip(relevant_factors, relevant_levels):
        for y, ylabel, max_plot_val, min_plot_val, boxplot in zip(ys, ylabels, max_plot_vals, min_plot_vals, boxplots):
            if len(can_be_dropped) == 0:
                plot_main(data, np.array([factor]), np.array([factor_levels]), y, 
                    ylabel, name, max_plot_val, boxplot, min_plot_val)
            else:
                plot_main_drop(data, np.array([factor]), np.array([factor_levels]), y, 
                    ylabel, name, can_be_dropped, max_plot_val, boxplot, min_plot_val)

    # Interactions
    print('Interaction effects')
    for factor, factor_levels in zip(relevant_factors, relevant_levels):
        print('Action: ' + factor)
        for y, ylabel, max_plot_val, min_plot_val, boxplot in zip(ys, ylabels, max_plot_vals, min_plot_vals, boxplots):
            print('Effects on ' + ylabel)
            plot_interactions(data, np.array([factor]), np.array([factor_levels]), 
                y, ylabel, name, contexts, context_levels, max_plot_val, boxplot, '', min_plot_val)
            for drop in can_be_dropped:
                temp = data[data[drop] == 1]
                plot_interactions(temp, np.array([factor]), np.array([factor_levels]), y, ylabel, name, contexts, 
                    context_levels, max_plot_val, boxplot, 'Dropped: ' + drop, min_plot_val)