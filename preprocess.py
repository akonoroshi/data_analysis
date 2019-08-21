import pandas as pd

def select_X(X: list, name: str) -> list:
    '''
    Return a list of independent variables relevant to an experiment specified by name.
    Hard-coded.
    '''
    if name == 'W10_insert':
        return ['motivate', 'metacognitive', 'friend']
    elif name == 'W10_bubble':
        return ['motivate', 'metacognitive', 'friend', 'big']
    elif name == 'W11_prepare':
        return ['motivate', 'metacognitive', 'friend', 'big', 'question', 'instructor']
    elif name == 'W12_prepare':
        return ['motivate', 'metacognitive', 'friend', 'big', 'question', 
            'instructor', 'sentence', 'research']
    elif name == 'W12_perform':
        return ['motivate', 'metacognitive', 'friend', 'big', 'question', 
            'sentence', 'research', 'how']
    return X

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