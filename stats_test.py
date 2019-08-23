import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportions_chisquare
import os
import subprocess
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import logit
from statsmodels.formula.api import probit
import itertools
import preprocess

def p_to_string(p: float) -> str:
    '''
    Generates a string saying the input p-value is less/greater than a critical value 
    (type I error rate) in statistics
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
    return returned

def ttest(data: pd.DataFrame, X: list, y: str, name: str, 
    old_exp=[], non_bin=np.empty((1,1)), alpha=0.05) -> dict:
    '''
    For each factor in X, conducts a t-test/one-way ANOVA to see if there are any 
    differences in the means of two or more groups.
    Parameters:
        data (pandas.DataFrame): df containing data of the experiment.
        X (list): the list of the independent variables.
        y (str): the name of the dependent variable.
        name (str): the version of the experiment.
        old_exp (list): the list of experiments that do not have any non-binary factors.
        non_bin (np.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each column in a raw represents a level.
        alpha (float): confidence level to reject the null hypotheses that the 
        dependent variable is normally distributed within each group and that the 
        variances are equal.
    Return:
        dict: keys are factors and values are the results of statistical tests. 
        "Problem" key is used as an index.
    '''
    result_dict = {'Problem': name.replace('_', ' ').title()}
    for x in X:
        if x in np.delete(non_bin, 0, axis=1): 
            # dummy variable of another factor
            continue
        elif name not in old_exp and x in non_bin[:,0]:
            # more than two levels
            base_data = data
            factor = np.where(non_bin[:,0] == x)[0][0]
            level_data = []
            for level in non_bin[factor]:
                base_data = base_data[base_data[level] == 0]
                level_data.append(data[data[level] == 1][y])
            level_data.append(base_data[y])
            
            try:
                # test normality
                normal = True
                for level in level_data:
                    normal = normal and stats.shapiro(level)[1] > alpha

                if normal: # normally distributed, so oneway anova
                    result = stats.f_oneway(*level_data)
                else: # not normally distributed, so Kruskal-Wallis test
                    result = stats.kruskal(*level_data)
            except ValueError as e:
                print('Not enough data for ' + x + ', so non-normality is assumed.')
                result = stats.kruskal(*level_data)
        else: # other factors have only two levels
            try:
                # test normality
                if stats.shapiro(data[data[x] == 1][y])[1] > alpha and stats.shapiro(
                    data[data[x] == 0][y])[1] > alpha: # normally distributed, so t-test
                    result = stats.ttest_ind(data[data[x] == 1][y], data[
                        data[x] == 0][y], equal_var = stats.bartlett(
                            data[data[x] == 1][y], data[data[x] == 0][y])[1] > alpha)
                else: # not normally distributed, so Mann-Whitney U-test
                    result = stats.mannwhitneyu(data[data[x] == 1][y], data[
                        data[x] == 0][y], alternative='two-sided')
            except ValueError as e:
                print('Not enough data for ' + x + ', so non-normality is assumed.')
                result = stats.mannwhitneyu(data[data[x] == 1][y], data[
                    data[x] == 0][y], alternative='two-sided')
            
        p = p_to_string(result[1])
        result_dict[x] = str(round(result[0], 2)) + ' ' + p
    return result_dict

def ztest(data: pd.DataFrame, X: list, y: str, name: str, 
    old_exp=[], non_bin=np.empty((1,1))) -> dict:
    '''
    For each factor in X, conducts a z-test/chi-squared test to see if there are any 
    differences in proportions.
    Parameters:
        data (pandas.DataFrame): df containing data of the experiment.
        X (list): the list of the independent variables.
        y (str): the name of the dependent variable.
        name (str): the version of the experiment.
        old_exp (list): the list of experiments that do not have any non-binary factors.
        non_bin (np.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each column in a raw represents a level.
    Return:
        dict: keys are factors and values are the results of statistical tests. 
        "Problem" key is used as an index.
    '''
    result_dict = {'Problem': name.replace('_', ' ').title()}
    filtered = data[data[y] == 1]

    for x in X:
        count = []
        nobs = []
        if x in np.delete(non_bin, 0, axis=1): 
            # dummy variable of another factor
            continue
        elif name not in old_exp and x in non_bin[:,0]:
            # more than levels so do a chi-squared test
            base_filtered = filtered
            base_data = data
            factor = np.where(non_bin[:,0] == x)[0][0]
            for level in non_bin[factor]:
                count.append(len(filtered[filtered[level] == 1]))
                nobs.append(len(data[data[level] == 1]))
                base_filtered = base_filtered[base_filtered[level] == 0]
                base_data = base_data[base_data[level] == 0]
            count.append(len(base_filtered))
            nobs.append(len(base_data))
            result = proportions_chisquare(np.array(count), np.array(nobs))
        else: # other factors have only two levels, hence z-test
            for i in range(1, -1, -1):
                count.append(len(filtered[filtered[x] == i]))
                nobs.append(len(data[data[x] == i]))
            result = proportions_ztest(np.array(count), np.array(nobs))

        p = p_to_string(result[1])
        result_dict[x] = str(round(result[0], 2)) + ' ' + p
    return result_dict

def latex_to_pdf(latex, file_name: str, file_loc: str, y: str):
    '''
    Converts a latex string to a .tex file and generate a .pdf table.
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

def test_to_tex(data: pd.DataFrame, data_dict: dict, X: str, y: str, file_loc: str, 
    test: str, old_exp=[], non_bin=np.empty((1,1))):
    '''
    Does statistical tests for each experiment and pooled data and converts the results 
    to a latex string.
    Only called within analyze function.
    Parameters:
        data (pandas.DataFrame): df containing overall data of the experiments.
        data_dict (dict): the dictionary whose key is the names of experiments and 
        value is data (pandas.DataFrame). If you have only one experiment, you can set 
        this as an empty dict.
        X (list): the list of the independent variables.
        y (str): the name of the dependent variable.
        file_loc (str): the path to the place where you want to save .tex files
        test (str): the name of test. Either ttest (if continuous) or ztest (if binary)
        old_exp (list): the list of experiments that do not have any non-binary factors.
        non_bin (np.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each column in a raw represents a level.
    '''
    df = pd.DataFrame()
    if test == 'ttest':
        for name, prob in data_dict.items():
            relevant_X = preprocess.select_X(X, name)
            df = df.append(ttest(prob, relevant_X, y, name, old_exp, non_bin), ignore_index=True)
        df = df.append(ttest(data, X, y, 'Overall', old_exp, non_bin), ignore_index=True)
    else:
        for name, prob in data_dict.items():
            df = df.append(ztest(prob, X, y, name, old_exp, non_bin), ignore_index=True)
        df = df.append(ztest(data, X, y, 'Overall', old_exp, non_bin), ignore_index=True)
    df = df.set_index('Problem')

    latex_to_pdf(df.to_latex(
        column_format='c'*(df.shape[1]+1), escape=False), test + '.tex', file_loc, y)

def generate_formula(X: list, y:str) -> str:
    '''
    Returns the formula of regression from a list of independent variables and 
    a dependent variable.
    '''
    formula = y + ' ~'
    for x in X: # Sub-group effects
        formula += ' + ' + x
    for c in itertools.combinations(X, 2): # Interaction effects
        formula += ' + ' + c[0] + ':' + c[1]

    # Motivate has three factors, and research is one of the dummy variable of that factor.
    formula = formula.replace(' + motivate:research', '')
    formula = formula.replace(' + research:motivate', '')
    # instructor was removed and how was added in week 12 perform
    formula = formula.replace(' + instructor:how', '')
    formula = formula.replace(' + how:instructor', '')
    return formula

def ols_anova(data, X: list, y: str, file_loc: str, name: str):
    '''
    Fits to OLS, conducts ANOVA, and generates latex strings of the results.
        Parameters:
        data (pandas.DataFrame): df containing data of the experiment.
        X (list): the list of the independent variables.
        y (str): the name of the dependent variable.
        file_loc (str): the path to the place where you want to save .tex files.
        name (str): the name of the experiment.
    '''
    formula = generate_formula(X, y)
    lm_all = ols(formula, data=data).fit()
    table = sm.stats.anova_lm(lm_all, typ=2)
    latex_to_pdf(table.to_latex(column_format='c'*(table.shape[1]+1), escape=False).replace(
        '_', ' ').replace('>', '$>$'), name + '_anova.tex', file_loc, y)
    latex_to_pdf('\end{adjustbox}\n\\begin{tabular}{lclc}'.join(
        lm_all.summary().as_latex().replace(r'\begin{tabular}{lcccccc}', 
            '\\begin{adjustbox}{width=1\\textwidth}\n\\begin{tabular}{lcccccc}').rsplit(
                r'\begin{tabular}{lclc}', 1)), name + '_ols.tex', file_loc, y)

def discrete_reg(data, X: list, y: str, file_loc: str, name: str, log=True):
    '''
    Fits to logistic or probit regression and generates a latex string of the result.
    Parameters:
        data (pandas.DataFrame): df containing data of the experiment.
        X (list): the list of the independent variables.
        y (str): the name of the dependent variable.
        file_loc (str): the path to the place where you want to save .tex files.
        name (str): the name of the experiment.
        log (bool): if True, fit to logistic regression; otherwise probit regression.
    '''
    formula = generate_formula(X, y)
    if log:
        lm_all = logit(formula, data=data).fit()
        suffix = '_logit.tex'
    else:
        lm_all = probit(formula, data=data).fit()
        suffix = '_probit.tex'
    latex_to_pdf('\end{tabular}\n\\end{adjustbox}'.join(
        lm_all.summary().as_latex().replace(r'\begin{tabular}{lcccccc}', 
            '\\begin{adjustbox}{width=1\\textwidth}\n\\begin{tabular}{lcccccc}').rsplit(
                r'\end{tabular}', 1)), name + suffix, file_loc, y)

def analyze(data: pd.DataFrame, data_dict: dict, X: list, y: str, 
    test: str, file_loc: str, log=True, old_exp=[], non_bin=np.empty((1,1))):
    '''
    Conducts a t-test/one-way ANOVA and fits to OLS if the dependent variable is continuous.
    Conducts a z-test/Chi-squared and fits to logistic (log=True) or probit (log=False) 
    refression if the dependent variable is binary. 
    Specify which test to conduct by test parameter.
    Parameters:
        data (pandas.DataFrame): df containing overall data of the experiments.
        data_dict (dict): the dictionary whose key is the names of experiments and 
        value is data (pandas.DataFrame). If you have only one experiment, you can set 
        this as an empty dict.
        X (list): the list of the independent variables.
        y (str): the name of the dependent variable.
        test (str): the name of test. Either ttest (if continuous) or ztest (if binary)
        file_loc (str): the path to the place where you want to save .tex files
        log (bool): if True, fit to logistic regression; otherwise probit regression.
        Ignored if test='ttest'.
        old_exp (list): the list of experiments that do not have any non-binary factors.
        non_bin (np.ndarray): the matrix of factor x level. Each raw represents a factor 
        and each column in a raw represents a level.
    '''
    test_to_tex(data, data_dict, X, y, file_loc, test, old_exp, non_bin)

    # For each experiment
    for name, prob in data_dict.items():
        relevant_X = preprocess.select_X(X, name)
        if test == 'ttest':
            ols_anova(prob, relevant_X, y, file_loc, name)
        else:
            discrete_reg(prob, relevant_X, y, file_loc, name, log)

    # Pooled results
    if test == 'ttest':
        ols_anova(data, X, y, file_loc, 'overall')
    else:
        discrete_reg(data, X, y, file_loc, 'overall', log)