# This file contains the command line tool such that given a movie in its
# name+name+year form it returns an integer estimate of how many times per day
# that movie is expected to be watched

import pickle
import pandas as pd
import requests
import json
from sklearn.ensemble import RandomForestRegressor


advanced_df = pd.read_csv('advanced_api_data.csv')

base_features = ['revenue', 'budget', 'vote_count', 'runtime',
                'other_prod_co', 'in_a_collection', 'popularity',
                'Adventure', 'Fantasy', 'other genre',
                'release_date', 'Animation', 'Thriller',
                'vote_average', 'Action', 'Science Fiction']

advanced_features = ['revenue', 'budget', 'vote_count', 'runtime',
                'other_prod_co', 'in_a_collection', 'popularity',
                'Adventure', 'Fantasy', 'other genre',
                'release_date', 'Animation', 'Thriller',
                'vote_average', 'Action', 'Science Fiction',
                'top100_director', 'metascore']


def preprocess(api):
    if len(api['belongs_to_collection']) > 0:
        api['in_a_collection'] = 1
    else:
        api['in_a_collection'] = 0
    del api['belongs_to_collection']

    genres = api['genres']

    # convert genres to one hot encoding
    api.update({'Action': 0})
    api.update({'Adult': 0})
    api.update({'Adventure': 0})
    api.update({'Animation': 0})
    api.update({'Biography': 0})
    api.update({'Comedy': 0})
    api.update({'Crime': 0})
    api.update({'Documentary': 0})
    api.update({'Drama': 0})
    api.update({'Family': 0})
    api.update({'Fantasy': 0})
    api.update({'Film': 0})
    api.update({'Noir': 0})
    api.update({'Game': 0})
    api.update({'Show': 0})
    api.update({'History': 0})
    api.update({'Horror': 0})
    api.update({'Musical': 0})
    api.update({'Music': 0})
    api.update({'Mystery': 0})
    api.update({'News': 0})
    api.update({'Romance': 0})
    api.update({'Science Fiction': 0})
    api.update({'Short': 0})
    api.update({'Sport': 0})
    api.update({'Thriller': 0})
    api.update({'War': 0})

    api.update({'Western': 0})
    api.update({'other genre': 0})

    for genre in genres:
        name = genre['name']
        if name in api:
            count = api[name]
            api.update({name: (count + 1)})
        else:
            count = api['other genre']
            api.update({'other genre': count + 1})

    del api['genres']

    # Clean out overview

    try:
        del api['overview']
    except KeyError:
        print(api)
    try:
        del api['poster_path']
    except KeyError:
        print(api)

    # convert production companies to one hot encoding
    pro_companies = api['production_companies']

    # most common production companies
    api.update({'New Line Cinema': 0})
    api.update({'Twentieth Century Fox Film Corporation': 0})
    api.update({'Miramax Films': 0})
    api.update({'TriStar Pictures': 0})
    api.update({'United Artists': 0})
    api.update({'Paramount Pictures': 0})
    api.update({'Columbia Pictures': 0})
    api.update({'Walt Disney Pictures': 0})
    api.update({'Warner Bros.': 0})
    api.update({'Metro-Goldwyn-Mayer (MGM)': 0})
    api.update({'Universal Pictures': 0})
    api.update({'Columbia Pictures Corporation': 0})
    api.update({'Touchstone Pictures': 0})
    api.update({'Canal+': 0})
    api.update({'other_prod_co': 0})

    for company in pro_companies:
        name = company['id']
        if name in api:
            count = api[name]
            api.update({name: (count + 1)})
        else:
            count = api['other_prod_co']
            api.update({'other_prod_co': count + 1})

    del api['production_companies']

    # convert spoken languages into one hot encoding
    spoken_languages = api['spoken_languages']

    # most common production companies
    api.update({'en': 0})
    api.update({'es': 0})
    api.update({'fr': 0})
    api.update({'it': 0})
    api.update({'de': 0})
    api.update({'other_spk_lng': 0})
    lgs = ['en', 'es', 'fr', 'it', 'de']

    for language in spoken_languages:
        name = language['iso_639_1']
        if name in lgs:
            count = api[name]
            api.update({name: count + 1})
        else:
            count = api['other_spk_lng']
            api.update({'other_spk_lng': count + 1})

    del api['spoken_languages']

    del api['production_countries']
    return api

#Convert all data to floats
def only_numeric(df):
    df = df.copy()
    df['release_date'] = df['release_date'].apply(lambda x: int(x[0:4]) + (int(x[5:7]) / 12))

    for item in df.keys():
        df[item] = df[item].apply(lambda x: float(x))

    return df

# Given a movie title will return an integer prediction of watches per day
# based on the models save in rf1.sav and advanced_rf1.sav
def predict(title):
    basic = pickle.load(open('rf1.sav', 'rb'))
    advanced = pickle.load(open('advanced_rf1.sav', 'rb'))

    url = 'http://128.2.204.215:8080/movie/' + title
    try:
        r = requests.get(url)
        api = pd.DataFrame([preprocess(r.json())])
    except (json.decoder.JSONDecodeError, KeyError):
        print("No movie matches the title:", title)
        return 0

    cut_down = api[base_features]
    floats = only_numeric(cut_down)

    row = advanced_df.loc[advanced_df['id'] == title]
    if(len(row) != 0):
        metascore = float(row['metascore'])
        top100_d = float(row['top100_director'])

        floats['metascore'] = metascore
        floats['top100_director'] = top100_d
        return int(advanced.predict(floats.to_numpy())[0])
    else:
        return int(basic.predict(floats.to_numpy())[0])


#read in title
title = input('Enter a movie id: ')
print(predict(title))

