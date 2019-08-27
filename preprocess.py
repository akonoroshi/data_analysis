import pandas as pd
import numpy as np

def select_factors(factors: np.ndarray, name: str) -> list:
    '''
    Return indices of independent variables relevant to an experiment specified by name.
    Hard-coded.
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

def add_filter(data, data_col: str, new_col: str):
    data[new_col] = 1
    data.loc[data[data_col] == -99, new_col] = 0
    data.loc[data[data_col].isna(), new_col] = 0
    return data

def filter_dict(data_dict: dict, y: str, threshold: int, smaller=True) -> dict:
    '''
    Filters data in data_dict so that data have only y values smaller/larger than 
    the threshold.
    '''
    filtered_dict = {}
    for name, data in data_dict.items():
        if smaller:
            filtered_dict[name] = data[data[y] < threshold]
        else:
            filtered_dict[name] = data[data[y] > threshold]
    return filtered_dict


def remove_personal_identifiers(data):
    personal_info = ['IPAddress', 'LocationLatitude', 'LocationLongitude', 
        'hashed_id', 'id', 'user_id', 'username']
    for info in personal_info:
        data[info] = ''
    return data