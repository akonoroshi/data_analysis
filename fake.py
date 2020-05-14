# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import pandas as pd
import plot
import stats_test
import numpy as np

#%%
# Read csv files
data = pd.read_csv('sample_data_preprocessed.csv')
pd.set_option('display.max_colwidth', 1000)

# Precondition: the length of factors and the number of rows of levels
factors = np.array(['message'])
# First levels are reference
# Must have at least two levels for each factor
levels = np.array([['1', '2']])

# Precondition: the length of contexts is the same as the number of rows of context_levels
contexts = np.array(['colour'])
# First column of context_levels is reference 
# Must have at least two levels for each context
context_levels = np.array([['red', 'green', 'blue']])

# Indicator columns if you want to drop optionally when plotting
can_be_dropped = []

#%% [markdown]
# # Analysis

#%%
ys = ['Q3', 'length'] # must be the same as column names
ylabels = ['Favourite Number', 'Length of String'] # must be the same length as ys
max_plot_vals = [100, 100]
min_plot_vals = [10, 10]
boxplots = [False, False]

#%% [markdown]
# ## Plots and Statistical Analysis

#%%
plot.explore_by_factor(data, factors, levels, ys, ylabels, 'Sample', can_be_dropped, max_plot_vals, min_plot_vals, boxplots, contexts, context_levels)

#%% [markdown]
# ### Favourite number

#%%
plot.explore(data, {}, factors, levels, 'Q3', 'Favourite Number', max_plot_val=100, contexts=contexts, context_levels=context_levels) # depreciated
#%%
stats_test.analyze(data, {}, factors, levels, 'Q3', 'ttest', './', contexts=contexts, context_levels=context_levels)

#%% [markdown]
# ### Length of strings

#%%
plot.explore(data, {}, factors, levels, 'length', 'Length of String', max_plot_val=100, contexts=contexts, context_levels=context_levels) # depreciated

#%%
stats_test.analyze(data, {}, factors, levels, 'length', 'ttest', './', contexts=contexts, context_levels=context_levels)
