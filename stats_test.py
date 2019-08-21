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

def ttest(data: pd.DataFrame, X: list, y: str, problem: str, equal_var=False) -> dict:
    '''
    For each factor in X, conducts a t-test/one-way ANOVA to see if there are any 
    differences in the means of two or more groups.
    Return:
        dict: keys are factors and values are the results of statistical tests. 
        "Problem" key is used as an index.
    '''
    result_dict = {'Problem': problem.replace('_', ' ').title()}
    for x in X:
        if x == 'research': # research is a dummy variable of factor motivate
            continue
        elif problem not in ['W10_insert', 'W10_bubble', 'W11_prepare'] and x == 'motivate':
            # motivate has three levels
            # test normality
            if stats.normaltest(data[(data[x] == 0) & (data['research'] == 0)][y])[
                1] > 0.05 and stats.normaltest(data[data[x] == 1][y])[1] > 0.05 and stats.normaltest(
                    data[data['research'] == 1][y])[1] > 0.05:
                # normally distributed, so oneway anova
                result = stats.f_oneway(data[(data[x] == 0) & (data['research'] == 0)][
                    y], data[data[x] == 1][y], data[data['research'] == 1][y])
            else:
                # not normally distributed, so Kruskal-Wallis test
                result = stats.kruskal(data[(data[x] == 0) & (data['research'] == 0)][
                    y], data[data[x] == 1][y], data[data['research'] == 1][y])
        else: # other factors have only two levels
            # test normality
            if stats.normaltest(data[data[x] == 0][y])[1] > 0.05 and stats.normaltest(
                data[data[x] == 1][y])[1] > 0.05: 
                # normally distributed, so t-test
                result = stats.ttest_ind(data[data[x] == 0][y], data[
                    data[x] == 1][y], equal_var = stats.bartlett(
                        data[data[x] == 0][y], data[data[x] == 1][y])[1] > 0.05)
            else:
                # not normally distributed, so Mann-Whitney U-test
                result = stats.mannwhitneyu(data[data[x] == 0][y], data[
                    data[x] == 1][y], alternative='two-sided')
            
        p = p_to_string(result[1])
        result_dict[x] = str(round(result[0], 2)) + ' ' + p
    return result_dict

def ztest(data: pd.DataFrame, X: list, y: str, problem: str) -> dict:
    '''
    For each factor in X, conducts a z-test/chi-squared test to see if there are any 
    differences in proportions.
    Return:
        dict: keys are factors and values are the results of statistical tests. 
        "Problem" key is used as an index.
    '''
    result_dict = {'Problem': problem.replace('_', ' ').title()}
    filtered = data[data[y] == 1]

    for x in X:
        if x == 'research': # research is a dummy variable of factor motivate
            continue
        elif problem not in ['W10_insert', 'W10_bubble', 'W11_prepare'] and x == 'motivate':
            # motivate has three levels so do a chi-squared test
            count = np.array((
                len(filtered[filtered[x] == 0]), len(filtered[filtered[x] == 1]), len(
                    filtered[filtered['research'] == 1])))
            nobs = np.array((len(data[data[x] == 0]), len(data[data[x] == 1]), len(
                data[data['research'] == 1])))
            result = proportions_chisquare(count, nobs)
        else: # other factors have only two levels, hence z-test
            count = np.array((
                len(filtered[filtered[x] == 0]), len(filtered[filtered[x] == 1])))
            nobs = np.array((len(data[data[x] == 0]), len(data[data[x] == 1])))
            result = proportions_ztest(count, nobs)

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

def test_to_tex(data: pd.DataFrame, data_dict: dict, X: str, y: str, file_loc: str, test: str):
    '''
    Does statistical tests for each experiment and pooled data and converts the results 
    to a latex string.
    Only called within analyze function.
    '''
    df = pd.DataFrame()
    if test == 'ttest':
        for name, prob in data_dict.items():
            relevant_X = preprocess.select_X(X, name)
            df = df.append(ttest(prob, relevant_X, y, name), ignore_index=True)
        df = df.append(ttest(data, X, y, 'Overall'), ignore_index=True)
    else:
        for name, prob in data_dict.items():
            df = df.append(ztest(prob, X, y, name), ignore_index=True)
        df = df.append(ztest(data, X, y, 'Overall'), ignore_index=True)
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

def logit_reg(data, X: list, y: str, file_loc: str, name: str):
    '''
    Fits to logistic regression and generates a latex string of the result.
    '''
    formula = generate_formula(X, y)
    lm_all = logit(formula, data=data).fit()
    latex_to_pdf('\end{tabular}\n\\end{adjustbox}'.join(
        lm_all.summary().as_latex().replace(r'\begin{tabular}{lcccccc}', 
            '\\begin{adjustbox}{width=1\\textwidth}\n\\begin{tabular}{lcccccc}').rsplit(
                r'\end{tabular}', 1)), name + '_logit.tex', file_loc, y)

def analyze(data: pd.DataFrame, data_dict: dict, X: list, y: str, 
    test: str, file_loc: str):
    '''
    Conducts a t-test/one-way ANOVA and fits to OLS if the dependent variable is continuous.
    Conducts a z-test/Chi-squared and fits to logistic refression if the dependent 
    variable is binary.
    Specify which test to conduct by test parameter.
    '''
    test_to_tex(data, data_dict, X, y, file_loc, test)

    # For each experiment
    for name, prob in data_dict.items():
        relevant_X = preprocess.select_X(X, name)
        if test == 'ttest':
            ols_anova(prob, relevant_X, y, file_loc, name)
        else:
            logit_reg(prob, relevant_X, y, file_loc, name)

    # Pooled results
    if test == 'ttest':
        ols_anova(data, X, y, file_loc, 'overall')
    else:
        logit_reg(data, X, y, file_loc, 'overall')