import numpy as np
import pandas as pd

def export(training, testing, model):
    '''
    inputs:
        - training: the training set of plays for the model
        - testing: the testing set of plays for the model
        - model: the model of interest
    output:
        - a pandas dataframe as dictated by Kaggle submission guidelines
        - https://www.kaggle.com/c/nfl-big-data-bowl-2020/overview/evaluation
    '''
    plays = [[
        f'{i} yards' for i in range(-99, 100, 1)
    ]]

    full_set = np.concatenate([training, testing])

    pred = [
        model.predict(np.array([instance,])) for instance in full_set
    ]

    for instance in pred:
        print(instance)
        yardage = []
        for i in range(-99, 100, 1):
            if i < instance[0]:
                yardage.append(0)
            else:
                yardage.append(1)
            plays.append(yardage)

    df = pd.DataFrame(plays)

    return df
