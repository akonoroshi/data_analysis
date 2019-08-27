# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import pandas as pd
import plot
import preprocess
import stats_test
import numpy as np

#%%
# Read csv files
data_op = pd.read_csv('W10_insert_fake.csv')
data_gr = pd.read_csv('W10_bubble_fake.csv')
data_11 = pd.read_csv('W11_prepare_fake.csv')
data_12_pre = pd.read_csv('W12_prepare_fake.csv')
data_12_pfm = pd.read_csv('W12_perform_fake.csv')
data = pd.read_csv('overall_fake.csv')
pd.set_option('display.max_colwidth', 1000)

#%%
# Preprocessing general to all experimental versions
data_list = [data_op, data_gr, data_11, data_12_pre, data_12_pfm]
data_names = ['W10_insert', 'W10_bubble', 'W11_prepare', 'W12_prepare', 'W12_perform']
data_dict = dict(zip(data_names, data_list))

# Precondition: the length of factors and the number of rows of levels
factors = np.array(['motivate', 'metacognitive', 'friend', 'big', 'question', 'instructor', 
    'sentence', 'how'])
# First levels are reference
# Must have at least two levels for each factor
levels = np.array([['no', 'stop', 'research'], ['no', 'yes'], ['no', 'yes'], ['no', 'yes'], [
    'no', 'yes'], ['no', 'yes'], ['no', 'yes'], ['no', 'yes']])
old_exp = ['W10_insert', 'W10_bubble', 'W11_prepare']

#%% [markdown]
# # Number of completed and partial responses

#%%
# Print the number of students assigned to each condition and 
# the number of students who completed the survey
for name, prob in data_dict.items():
    print(name.replace('_', ' ').title())
    indices = preprocess.select_factors(factors, name)
    for i in indices:
        message = factors[i] + ': '
        if len(levels[i]) != 2 and name not in old_exp:
            base_data = prob
            for j in range(1, len(levels[i])):
                message += '{0} received "{1}" ({2} completed) vs '.format(len(prob[prob[
                    factors[i] + '_' + levels[i][j]] == 1]), levels[i][j], len(prob[(prob[
                        factors[i] + '_' + levels[i][j]] == 1) & (prob['Finished'] == 1)]))
                base_data = base_data[base_data[factors[i] + '_' + levels[i][j]] == 0]
            message += '{0} received none ({1} completed)'.format(len(base_data), len(base_data[
                base_data['Finished'] == 1]))
        else:
            message += '{0} assigned ({1} completed) vs {2} not assigned ({3} completed)'.format(
                len(prob[prob[factors[i] + '_' + levels[i][1]] == 1]), len(prob[(prob[
                    factors[i] + '_' + levels[i][1]] == 1) & (prob['Finished'] == 1)]), len(prob[prob[
                        factors[i] + '_' + levels[i][1]] == 0]), len(prob[(prob[factors[
                            i] + '_' + levels[i][1]] == 0) & (prob['Finished'] == 1)]))
        print(message)

#%% [markdown]
# # Statistical Analysis

#%% [markdown]
# ## Self-reported helpfulness

#%%
plot.explore(data, data_dict, factors, levels, 'expl_helpfulness', 
    'Self-reported helpfulness', max_plot_val=4, old_exp=old_exp)
#%%
stats_test.analyze(data, data_dict, factors, levels, 'expl_helpfulness', 'ttest', './', 
    old_exp=old_exp)

#%% [markdown]
# ## Correct rate on a related problem

#%%
plot.explore(data, data_dict, factors, levels, 'correct_rate', 
    'Correct rate on related problem', old_exp=old_exp)

#%%
stats_test.analyze(data, data_dict, factors, levels, 'correct_rate', 'ttest', './', 
    old_exp=old_exp)

#%% [markdown]
# ## Time spent on answering the survey 

#%% 
plot.explore(data, data_dict, factors, levels, 'Duration', 'Duration', 
    max_plot_val=500, boxplot=True, old_exp=old_exp)

#%%
# TODO should I exclude outliers?
#threshold = data['Duration'].quantile(0.75) * 2
#filtered = data[data['Duration'] < threshold]
#filtered_dict = preprocess.filter_dict(data_dict, 'Duration', threshold)
#stats_test.analyze(filtered, filtered_dict, factors, levels, 'Duration', 'ttest', './', old_exp=old_exp)
stats_test.analyze(data, data_dict, factors, levels, 'Duration', 'ttest', 
    './', old_exp=old_exp)

#%% [markdown]
# ## Quality (length) of explanation
#%% 
# Remove students who didn't write any explanations
threshold = 1
filtered = data[data['explanation_length'] > threshold]
filtered_dict = preprocess.filter_dict(data_dict, 'explanation_length', threshold, smaller=False)
plot.explore(filtered, filtered_dict, factors, levels, 'explanation_length', 
    'Length of explanations', max_plot_val=400, boxplot=True, old_exp=old_exp)

#%%
# TODO should I exclude outliers?
#threshold = data['explanation_length'].quantile(0.75) * 2
#filtered = data[data['explanation_length'] < threshold]
#filtered_dict = preprocess.filter_dict(data_dict, 'explanation_length', threshold)
stats_test.analyze(filtered, filtered_dict, factors, levels, 'explanation_length', 
    'ttest', './', old_exp=old_exp)
#stats_test.analyze(data, data_dict, factors, levels, 'explanation_length', 'ttest', './', old_exp=old_exp)

#%% [markdown]
# ## Proportion of people who wrote explanations
#%% 
plot.explore(data, data_dict, factors, levels, 'has_explanation', 'Has explanation', 
    max_plot_val=0.5, old_exp=old_exp)

#%%
stats_test.analyze(data, data_dict, factors, levels, 'has_explanation', 'ztest', './', 
    old_exp=old_exp, log=False)

#%% [markdown]
# ## Proportion of people who completed the survey
#%%
plot.explore(data, data_dict, factors, levels, 'Finished', 'Completion rate', max_plot_val=0.6, 
    old_exp=old_exp)

#%%
stats_test.analyze(data, data_dict, factors, levels, 'Finished', 'ztest', './', 
    old_exp=old_exp)
