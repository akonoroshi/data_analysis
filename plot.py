import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import preprocess

def discrete_hist(data, y: str, min_val: int, max_val: int, step: float, title: str):
    plt.hist(data[y], bins=np.arange(min_val - step/2, max_val + step/2, step))
    plt.title(title)
    plt.xticks(range(min_val, max_val))
    plt.show()

def build_plot(ax, data: pd.DataFrame, factor: str, factor_levels: np.ndarray, y: str,
    name: str, title: str, max_plot_val=1.0, boxplot=False, old_exp=[]):
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
    if name in old_exp: # factors with two levels
        groups = ['no', 'yes']
        mean_outcomes = mean_outcomes[:2]
        num_students = num_students[:2]
        if not boxplot:
            se_outcomes = se_outcomes[:2]
            
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
            ax.text(i+1, 0.2*max_plot_val, 'Median =', ha='center', va='bottom',
                    fontweight='bold', fontsize = 16)
            ax.text(i+1, 0.15*max_plot_val, str(mean_outcomes[
                i].median()), ha='center', va='bottom', fontweight='bold', fontsize = 16)
            ax.text(i+1, 0.05*max_plot_val, 'n = %s' %num_students[
                i], ha='center', va='bottom', fontweight='bold', fontsize = 16)
        else:
            ax.text(i, 0.2*max_plot_val, 'Mean =', ha='center', va='bottom', fontweight='bold', 
                fontsize = 14)
            ax.text(i, 0.15*max_plot_val, str(np.round(
                mean_outcomes[i],2)), ha='center', va='bottom', fontweight='bold', 
                fontsize = 14)
            ax.text(i, 0.05*max_plot_val, 'n = %s' %num_students[
                i], ha='center', va='bottom', fontweight='bold', fontsize = 14)

    ax.set_title(title, fontsize = 16)
    ax.set_ylim(0, max_plot_val)
    ax.yaxis.grid(True)

def plot_interactions(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, 
    y: str, label: str, name: str, max_plot_val=1.0, boxplot=False, old_exp=[]):
    '''
    For each valid combination of independent (dummy) variables, plots interaction effects.
    Parameters:
        data (pandas.DataFrame): df containing data of the experiment.
        factors (numpy.ndarray): the list of the independent variables.
        levels (numpy.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each element in a raw represents a level.
        y (str): the name of the dependent variable.
        label (str): the description of the dependent variable that goes to a y-axis of a figure.
        name (str): the version of the experiment.
        boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
        old_exp (list): the list of experiments that do not have any non-binary factors.
    '''

    for c in itertools.combinations(range(len(factors)), 2): 
        fig, ax = plt.subplots(1, len(levels[c[1]]), figsize=(8,8))
        ax = ax.ravel()
        base_data = data # reference level
        for i in range(1, len(levels[c[1]])):
            base_data = base_data[
                base_data[factors[c[1]] + '_' + levels[c[1]][i]] == 0]
            filtered = data[data[factors[c[1]] + '_' + levels[c[1]][i]] == 1]
            try:
                build_plot(ax[i], filtered, factors[c[0]], levels[c[0]], y, name, 
                    levels[c[1]][i], max_plot_val, boxplot, old_exp)
            except ValueError as e:
                print('Not enough data for ' + factors[c[0]] + ' and ' + factors[c[1]])
                continue
        try:
            build_plot(ax[0], base_data, factors[c[0]], levels[c[0]], y, name, 
                levels[c[1]][0], max_plot_val, boxplot, old_exp)
        except ValueError as e:
            print('Not enough data for ' + factors[c[0]] + ' and ' + factors[c[1]])
        finally:   
            # Save the figure and show
            fig.suptitle(name.replace('_', ' ').title() + ': ' + factors[c[1]], fontsize = 18)
            fig.text(0.5, 0.04, factors[c[0]], ha='center', fontsize = 16)
            fig.text(0.04, 0.5, label, va='center', rotation='vertical', fontsize = 16)
            plt.show()
        
def plot_main(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, y: str, 
    label: str, name: str, max_plot_val=1.0, boxplot=False, old_exp=[]):
    '''
    For each valid combination of independent (dummy) variables, plots interaction effects.
    Parameters:
        data (pandas.DataFrame): df containing data of the experiment.
        factors (numpy.ndarray): the list of the independent variables.
        levels (numpy.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each element in a raw represents a level.
        y (str): the name of the dependent variable.
        label (str): the description of the dependent variable that goes to a y-axis of a figure.
        name (str): the version of the experiment.
        boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
        old_exp (list): the list of experiments that do not have any non-binary factors.
    '''
    
    for factor, factor_levels in zip(factors, levels):
        fig, ax = plt.subplots()
        try:
            build_plot(ax, data, factor, factor_levels, y, name, name.replace(
                '_', ' ').title(), max_plot_val, boxplot, old_exp)
        except ValueError as e:
            print('Not enough data for ' + factor)
            continue
        ax.set_ylabel(label, fontsize = 16)
        ax.set_xlabel(factor, fontsize = 16)
   
        # Save the figure and show
        plt.show()

def explore(data: pd.DataFrame, data_dict: dict, factors: np.ndarray, 
    levels: np.ndarray, y: str, ylabel: str, max_plot_val=1.0, 
        boxplot=False, old_exp=[]):
    '''
    Plots main effects and interaction effects for each experiment and the pooled result.
    Parameters:
        data (pandas.DataFrame): df containing overall data of the experiments
        data_dict (dict): the dictionary whose key is the names of experiments and 
        value is data (pandas.DataFrame). If you have only one experiment, you can set 
        this as an empty dict.
        factors (numpy.ndarray): the list of the independent variables.
        levels (numpy.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each element in a raw represents a level.
        y (str): the name of the dependent variable.
        ylabel (str): the description of y that appears in graphs
        boxplot (bool): True if you want to use boxplots. False if you want bar graphs.
        old_exp (list): the list of experiments that do not have any non-binary factors.
    '''
    # For each experiment
    for name, prob in data_dict.items():
        indices = preprocess.select_factors(factors, name)
        relevant_factors = factors[indices]
        relevant_levels = levels[indices]
        plot_main(prob, relevant_factors, relevant_levels, y, ylabel, 
            name, max_plot_val, boxplot, old_exp)
        plot_interactions(prob, relevant_factors, relevant_levels, y, ylabel, 
            name, max_plot_val, boxplot, old_exp)

    # Pooled result
    plot_main(data, factors, levels, y, ylabel, 'Overall', 
        max_plot_val, boxplot, old_exp)
    plot_interactions(data, factors, levels, y, ylabel, 'Overall', 
        max_plot_val, boxplot, old_exp)