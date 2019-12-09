import pandas as pd
import numpy as np
from datetime import date
from scipy.special import expit 


def encode(data, column):
    sets = list(set(data[column]))
    return [int(sets.index(temp)) for temp in data[column]]


def encode_game_time(data, column):

    def classify(h, m, s, quarter):
        if h in [0, 1, 2, 3, 5, 6]:
            return 2
        elif h in [19, 20, 21, 22, 23]:
            return 1
        else:
            return 0

    date_time = [data.split('T') for data in data[column]]
    time = [entry[1].split('.') for entry in date_time]
    time = [entry[0] for entry in time]
    classification = [
        classify(
            int(entry[0]), int(entry[1]), int(entry[2]), int(quarter)
        ) for entry, quarter in zip(
            (entry.split(':') for entry in time), data['quarter']
        )
    ]
    return classification


def encode_wind_speed(data):
    '''
    TODO: encode the windspeed cols that are not filled with the average for all speeds (Regularization)
    '''
    temp = []
    for item in data:
        if type(item) == str:
            string = item.split(' ')
            temp.append(string[0])
        else:
            temp.append(item)
    return temp


def run(path):
    data = pd.read_csv(path, header='infer')

    # convert column names to lower case
    data.columns = data.columns.map(lambda x: x.lower())
    
    # convert UTC time of snap to gametime classification
    for col in ['timesnap', 'timehandoff']:
        data[col] = encode_game_time(data, col)

    data = data.drop(columns=['windspeed', 'winddirection', 'temperature', 'gameweather', 'stadiumtype'])
    
    # values which we will map
    data['team'] = data['team'].map({'away':0, 'home':1})
    data['fieldposition'] = [0 if pos == home else 1 for pos, home in zip(
        data['fieldposition'], data['hometeamabbr']
    )]

    # columns that will be encoded as integers based on set
    columns = [
        'offenseformation','offensepersonnel','defensepersonnel','playdirection',
        'position', 'displayname', 'playercollegename', 'stadium', 'location',
        'hometeamabbr', 'visitorteamabbr', 'turf', 'possessionteam'
    ]

    for i in range(len(columns)): 
        data[columns[i]] = encode(data, columns[i])

    # date columns that will be converted integer values
    data['gameclock'] = data['gameclock'].str.split(':')
    data['gameclock'] = [60*int(lyst[0]) + int(lyst[1]) for lyst in data['gameclock']]

    # encode height
    data['playerheight'] = data['playerheight'].str.split('-')
    data['playerheight'] = [12*int(item[0])+int(item[1]) for item in data['playerheight']]

    # encode birth date -- days since birth date
    data['playerbirthdate'] = [
        (date.today() - date(
            int(bd[2]), int(bd[0]), int(bd[1])
        )).days for bd in (
            birthdate.split('/')  for birthdate in data['playerbirthdate']
        )
    ]

    # for col in data.columns:
    #    data[col] = expit(data[col])

    # flatten rows in dataframe
    return np.array([row.values.flatten() for index, row in data.iterrows()], dtype='float32'), data['yards']
