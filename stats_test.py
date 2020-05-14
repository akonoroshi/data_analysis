import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportions_chisquare
import os
import subprocess
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.formula.api import ols
from statsmodels.formula.api import logit
from statsmodels.formula.api import probit
import itertools
import preprocess

pd.set_option('display.max_colwidth', 1000)

def correlations(x, y):
    print("Pearson's R: " + str(stats.pearsonr(x, y)))
    print("Spearman's Rho: " + str(stats.spearmanr(x, y)))
    
def p_to_string(p: float) -> str:
    '''
    Generates a string saying the input p-value is less/greater than a critical value 
    (type I error rate) in statistics.
    Parameters
    ----------
    p (float): the p-value from statistical test.

    Return
    -------
    returned (str): a latex string saying p is less/greater than a critical value.
    '''
    returned = ''
    if p < 0.01:
        returned = '$(p<0.01)$'
    elif p < 0.05:
        returned = '$(p<0.05)$'
    elif p < 0.1:
        returned = '$(p<0.1)$'
    elif p < 0.5:
        returned = '$(p>0.1)$'
    elif p <= 1:
        returned = '$(p>0.5)$'
    #return returned
    return '$(p=' + str(np.round(p, 2)) + ')$'

def non_normal(level_data: list):
    '''
    Returns the results of non-parametric testing. Either Mann-Whitney rank test or 
    Kruskal-Wallis H-test.
    Parameters
    ----------
    level_data (list): the list of pandas.Series that contains samples you want to
    run a hypothesis test on.

    Return
    -------
    statistic (float): U statistic or H statistic.
    p-value (float): the p-value.
    '''
    if len(level_data) == 2:
        return stats.mannwhitneyu(
            level_data[0], level_data[1], alternative='two-sided')
    else:
        return stats.kruskal(*level_data)

def ttest(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, y: str, 
    name: str, alpha=0.05) -> dict:
    '''
    For each factor, conducts a t-test/one-way ANOVA to see if there are any 
    differences in the means of two or more groups.
    Parameters
    ----------
    data (pandas.DataFrame): df containing data of the experiment.
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each row represents a 
    factor and each element in a row represents a level.
    y (str): the name of the dependent variable.
    name (str): the version of the experiment.
    alpha (float): confidence level to reject the null hypotheses that the 
    dependent variable is normally distributed within each group and that the 
    variances are equal.

    Return
    -------
    result_dict (dict): keys are factors and values are the results of statistical tests. 
    "Experiment" key is used as an index.
    '''
    result_dict = {'Experiment': name.replace('_', ' ').title()}
    for factor, factor_levels in zip(factors, levels):
        base_data = data
        level_data = []
        for i in range(1, len(factor_levels)):
            base_data = base_data[base_data[factor + '_' + factor_levels[i]] == 0]
            level_data.append(data[data[factor + '_' + factor_levels[i]] == 1][y])
        level_data.append(base_data[y])
            
        try:
            # test normality
            normal = True
            for level in level_data:
                normal = normal and stats.shapiro(level)[1] > alpha

            if normal: # normally distributed, so t-test or oneway anova
                if len(level_data) == 2:
                    result = stats.ttest_ind(level_data[0], level_data[
                        1], equal_var = stats.bartlett(*level_data)[1] > alpha)
                else:
                    result = stats.f_oneway(*level_data)
            else: # not normally distributed, so Kruskal-Wallis test
                result = non_normal(level_data)
        except ValueError as e:
            print('Not enough data for ' + factor + ', so non-normality is assumed.')
            result = non_normal(level_data)
            
        p = p_to_string(result[1])
        result_dict[factor] = str(round(result[0], 2)) + ' ' + p
    return result_dict

def ztest(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, y: str, 
    name: str, alpha=0.05) -> dict:
    '''
    For each factor in factors, conducts a t-test/one-way ANOVA to see if there are any 
    differences in the means of two or more groups.
    Parameters
    ----------
    data (pandas.DataFrame): df containing data of the experiment.
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each row represents a 
    factor and each element in a row represents a level.
    y (str): the name of the dependent variable.
    name (str): the version of the experiment.
    alpha (float): ignored.

    Return
    -------
    result_dict (dict): keys are factors and values are the results of statistical tests. 
    "Experiment" key is used as an index.
    '''
    result_dict = {'Experiment': name.replace('_', ' ').title()}
    filtered = data[data[y] == 1]

    for factor, factor_levels in zip(factors, levels):
        count = []
        nobs = []
        base_data = data
        base_filtered = filtered
        for i in range(1, len(factor_levels)):
            base_data = base_data[base_data[factor + '_' + factor_levels[i]] == 0]
            base_filtered = base_filtered[base_filtered[
                factor + '_' + factor_levels[i]] == 0]
            count.append(len(filtered[filtered[
                factor + '_' + factor_levels[i]] == 1]))
            nobs.append(len(data[data[factor + '_' + factor_levels[i]] == 1]))
        count.append(len(base_filtered))
        nobs.append(len(base_data))
        
        if len(count) > 2:
            result = proportions_chisquare(np.array(count), np.array(nobs))
        else:
            result = proportions_ztest(np.array(count), np.array(nobs))

        p = p_to_string(result[1])
        result_dict[factor] = str(round(result[0], 2)) + ' ' + p
    return result_dict

def latex_to_pdf(latex, file_name: str, file_loc: str, y: str):
    '''
    Converts a latex string to a .tex file and generate a .pdf table.
    Parameters
    ----------
    latex: latex string
    file_name (str): the name of the output latex file
    file_loc (str): the path to the place where you want to save .tex files
    y (str): the name of the dependent variable.
    '''
    template = r'''\documentclass[preview]{{standalone}}
\usepackage{{booktabs}}
\usepackage{{makecell}}
\usepackage{{adjustbox}}
\begin{{document}}
{}\end{{document}}
'''
    loc = file_loc + 'tables/' + y + '/'
    if not os.path.exists(loc):
        os.makedirs(loc)
        
    with open(loc + file_name, 'wb') as f:
        f.write(bytes(template.format(latex),'UTF-8'))
    subprocess.call(['pdflatex', file_name], cwd=loc)

def test_to_tex(data: pd.DataFrame, data_dict: dict, factors: np.ndarray, 
    levels: np.ndarray, y: str, file_loc: str, test: str):
    '''
    Does statistical tests for each experiment and pooled data and converts the results 
    to a latex string.
    Only called within analyze function.
    Parameters
    ----------
    data (pandas.DataFrame): df containing overall data of the experiments.
    data_dict (dict): the dictionary whose key is the names of experiments and 
    value is data (pandas.DataFrame). If you have only one experiment, you can set 
    this as an empty dict.
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each row represents a 
    factor and each element in a row represents a level.
    y (str): the name of the dependent variable.
    file_loc (str): the path to the place where you want to save .tex files
    test (str): the name of test. Either ttest (if continuous) or ztest (if binary)
    '''

    tests_dict = {'ttest': ttest, 'ztest': ztest}
    try:
        test_func = tests_dict[test]
    except KeyError as e:
        raise ValueError('Unsupprted Test')
    
    df = pd.DataFrame()
    for name, prob in data_dict.items():
        indices = preprocess.select_factors(factors, name)
        relevant_factors = factors[indices]
        relevant_levels = levels[indices]
        df = df.append(test_func(prob, relevant_factors, relevant_levels, y, name), ignore_index=True)
    df = df.append(test_func(data, factors, levels, y, 'Overall'), ignore_index=True)

    df = df.set_index('Experiment')
    print(df)

    latex_to_pdf(df.to_latex(
        column_format='c'*(df.shape[1]+1), escape=False), test + '.tex', file_loc, y)

def generate_formula(factors: np.ndarray, levels: np.ndarray, y:str, name: str, contexts=[], context_levels=[[]]) -> str:
    '''
    Returns the formula of regression from a list of independent variables and 
    a dependent variable.
    Parameters
    ----------
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each row represents a factor 
    and each element in a row represents a level.
    y (str): the name of the dependent variable.
    name (str): the name of the experiment.
    contexts (numpy.ndarray): the list of contextual variables. Set to be None if you 
    want to plot the interactions between action variables.
    context_levels (numpy.ndarray): the matrix of context x level. Each row represents a context 
    and each element in a row represents a level.

    Return
    -------
    formula (str): R-like formula used by statsmodels. Looks like y ~ x.
    '''
    formula = y + ' ~'
    for factor, factor_levels in zip(factors, levels): # Sub-group effects
        for i in range(1, len(factor_levels)):
            formula += ' + ' + factor + '_' + factor_levels[i]
    for context, context_level in zip(contexts, context_levels):
        for i in range(1, len(context_level)):
            formula += ' + ' + context + '_' + context_level[i] 

    if len(contexts) != 0:
        for factor, factor_levels in zip(factors, levels):
            for i in range(1, len(factor_levels)):
                for context, context_level in zip(contexts, context_levels):
                    for j in range(1, len(context_level)):
                        formula += ' + ' + factor + '_' + factor_levels[
                            i] + ':' + context + '_' + context_level[j]
    else:
        for c in itertools.combinations(range(len(factors)), 2): # Interaction effects
            for i in range(1, len(levels[c[0]])):
                for j in range(1, len(levels[c[1]])):
                    formula += ' + ' + factors[c[0]] + '_' + levels[c[0]][
                        i] + ':' + factors[c[1]] + '_' + levels[c[1]][j]

    return formula

def ols_anova(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, y: str, 
    file_loc: str, name: str, contexts=[], context_levels=[[]]):
    '''
    Fits to OLS, conducts ANOVA, and generates latex strings of the results.
    Parameters
    ----------
    data (pandas.DataFrame): df containing data of the experiment.
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each row represents a factor 
    and each element in a row represents a level.
    y (str): the name of the dependent variable.
    file_loc (str): the path to the place where you want to save .tex files.
    name (str): the name of the experiment.
    log (bool): if True, fit to logistic regression; otherwise probit regression.
    contexts (numpy.ndarray): the list of contextual variables. Set to be None if you 
    want to plot the interactions between action variables.
    context_levels (numpy.ndarray): the matrix of context x level. Each row represents a context 
    and each element in a row represents a level.
    '''
    print(name)
    formula = generate_formula(factors, levels, y, name, contexts, context_levels)
    lm_all = ols(formula, data=data).fit()
    print(lm_all.summary())
    table = sm.stats.anova_lm(lm_all, typ=2)
    print(table)
    latex_to_pdf(table.to_latex(column_format='c'*(table.shape[1]+1), escape=False).replace(
        '_', ' ').replace('>', '$>$'), name + '_anova.tex', file_loc, y)
    latex_to_pdf('\end{adjustbox}\n\\begin{tabular}{lclc}'.join(
        lm_all.summary().as_latex().replace(r'\begin{tabular}{lcccccc}', 
            '\\begin{adjustbox}{width=1\\textwidth}\n\\begin{tabular}{lcccccc}').rsplit(
                r'\begin{tabular}{lclc}', 1)), name + '_ols.tex', file_loc, y)

def discrete_reg(data: pd.DataFrame, factors: np.ndarray, levels: np.ndarray, y: str, 
    file_loc: str, name: str, log=True, contexts=[], context_levels=[[]], regularize=False):
    '''
    Fits to logistic or probit regression and generates a latex string of the result.
    Parameters
    ----------
    data (pandas.DataFrame): df containing data of the experiment.
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each row represents a factor 
    and each element in a row represents a level.
    y (str): the name of the dependent variable.
    file_loc (str): the path to the place where you want to save .tex files.
    name (str): the name of the experiment.
    log (bool): if True, fit to logistic regression; otherwise probit regression.
    contexts (numpy.ndarray): the list of contextual variables. Set to be None if you 
    want to plot the interactions between action variables.
    context_levels (numpy.ndarray): the matrix of context x level. Each row represents a context 
    and each element in a row represents a level.
    '''
    print(name)
    formula = generate_formula(factors, levels, y, name, contexts, context_levels)
    
    if log:
        if regularize:
            lm_all = glm(formula=formula, data=data, family=sm.families.Binomial()).fit_regularized(L1_wt=0.0)
        else:
            lm_all = logit(formula, data=data).fit(method='bfgs')
        suffix = '_logit.tex'
    else:
        lm_all = probit(formula, data=data).fit(method='bfgs')
        suffix = '_probit.tex'
    if regularize:
        print(lm_all.params)
    else:
        print(lm_all.summary())
        latex_to_pdf('\end{tabular}\n\\end{adjustbox}'.join(
            lm_all.summary().as_latex().replace(r'\begin{tabular}{lcccccc}', 
                '\\begin{adjustbox}{width=1\\textwidth}\n\\begin{tabular}{lcccccc}').rsplit(
                    r'\end{tabular}', 1)), name + suffix, file_loc, y)

def analyze(data: pd.DataFrame, data_dict: dict, factors: np.ndarray, levels: np.ndarray, y: str, test: str, file_loc: str, 
    log=True, contexts=[], context_levels=[[]], regularize=False):
    '''
    Conducts a t-test/one-way ANOVA and fits to OLS if the dependent variable is continuous.
    Conducts a z-test/Chi-squared and fits to logistic (log=True) or probit (log=False) 
    refression if the dependent variable is binary. 
    Specify which test to conduct by test parameter.
    Parameters
    ----------
    data (pandas.DataFrame): df containing overall data of the experiments.
    data_dict (dict): the dictionary whose key is the names of experiments and 
    value is data (pandas.DataFrame). If you have only one experiment, you can set 
    this as an empty dict.
    factors (numpy.ndarray): the list of the independent variables.
    levels (numpy.ndarray): the matrix of factor x level. Each raw represents a factor 
    and each column in a raw represents a level.
    y (str): the name of the dependent variable.
    test (str): the name of test. Either ttest (if continuous) or ztest (if binary)
    file_loc (str): the path to the place where you want to save .tex files
    log (bool): if True, fit to logistic regression; otherwise probit regression.
    Ignored if test='ttest'.
    contexts (numpy.ndarray): the list of contextual variables. Set to be None if you 
    want to plot the interactions between action variables.
    context_levels (numpy.ndarray): the matrix of context x level. Each row represents a context 
    and each element in a row represents a level.
    '''

    test_to_tex(data, data_dict, factors, levels, y, file_loc, test)

    # For each experiment
    for name, prob in data_dict.items():
        indices = preprocess.select_factors(factors, name)
        relevant_factors = factors[indices]
        relevant_levels = levels[indices]
        if test == 'ttest':
            ols_anova(prob, relevant_factors, relevant_levels, y, file_loc, name, contexts, context_levels)
        else:
            discrete_reg(prob, relevant_factors, relevant_levels, y, file_loc, name, log, contexts, context_levels, regularize)

    # Pooled results
    if test == 'ttest':
        ols_anova(data, factors, levels, y, file_loc, 'overall', contexts, context_levels)
    else:
        discrete_reg(data, factors, levels, y, file_loc, 'overall', log, contexts, context_levels, regularize)