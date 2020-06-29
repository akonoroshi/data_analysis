import pandas as pd
import numpy as np
import urllib.request
import json

def select_factors(factors: np.ndarray, name: str) -> list:
    '''
    Returns indices of independent variables relevant to an experiment specified by name.
    Hard-coded.
    Parameters
    ----------
    factors (numpy.ndarray): the list of the independent variables.
    name (str): the version of the experiment.

    Return
    -------
    (list): a list of indices of relevant independent variables
    '''
    if name == 'W10_insert':
        return [0, 1, 2]
    elif name == 'W10_bubble':
        return [0, 1, 2, 3]
    elif name == 'W11_prepare':
        return [0, 1, 2, 3, 4, 5]
    elif name == 'W12_prepare':
        return [0, 1, 2, 3, 4, 5, 6]
    elif name == 'W12_perform':
        return [0, 1, 2, 3, 4, 6, 7]
    return list(range(len(factors)))

def add_filter(data: pd.DataFrame, data_col: str, new_col: str) -> pd.DataFrame:
    '''
    Add a column that indicates data is missing.
    Parameters
    ----------
    data (pandas.DataFrame): df containing overall data of the experiments.
    data_col (str): the column with possible missing values.
    new_col (str): the name of new indicator column.

    Return
    -------
    data (pandas.DataFrame): df containing overall data of the experiments 
    with a new indicator column.
    '''
    data[new_col] = 1
    data.loc[data[data_col] == -99, new_col] = 0
    data.loc[data[data_col].isna(), new_col] = 0
    return data

def filter_dict(data_dict: dict, y: str, threshold, smaller=True) -> dict:
    '''
    Filters data in data_dict so that data have only y values smaller/larger than 
    the threshold.
    data_dict (dict): the dictionary whose key is the names of experiments and 
    value is data (pandas.DataFrame).
    y (str): the name of the dependent variable.
    threshold (int or float): threshold.
    smaller (bool): True if you want to keep data smaller than threshold. Otherwise False.

    Return
    -------
    filtered_dict (dict): a dictionary whose key is the names of experiments and 
    value is data (pandas.DataFrame) after dropping some data according to threshold.
    '''
    filtered_dict = {}
    for name, data in data_dict.items():
        if smaller:
            filtered_dict[name] = data[data[y] < threshold]
        else:
            filtered_dict[name] = data[data[y] > threshold]
    return filtered_dict

def remove_personal_identifiers(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Removes personal information from data. Hard-coded.
    Parameters
    ----------
    data (pandas.DataFrame): df containing overall data of the experiments

    Return
    -------
    data (pandas.DataFrame): df containing overall data of the experiments 
    without personal infomation.
    '''

    personal_info = ['IPAddress', 'LocationLatitude_x', 'LocationLongitude_x', 
        'id', 'user_id', 'username']
    for info in personal_info:
        data[info] = ''
    return data

def get_velues_from_mooclets(url: str, token: str) -> pd.DataFrame:
    '''
    Get values from mooclets.
    Parameters
    ----------
    url (str): string of url of the mooclets engine. The format parameter must be 
    equal to json.
    token (str): token for authorization

    Return
    -------
    values (pandas.DataFrame): df containing data related to each value
    '''
    header = {'Authorization': 'Token ' + token}
    values = []
    cur_url = url
    while cur_url != None:
        req = urllib.request.Request(cur_url, headers=header)
        response = urllib.request.urlopen(req)
        data_json = json.loads(response.read().decode())
        cur_url = data_json['next']
        values += data_json['results']

    return pd.DataFrame(values)