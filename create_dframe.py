import json
import requests
import csv


f = open("movie_titles.txt", "r")
fl = f.readlines()
f.close()

# Transform additional csv dataframe(s) into searchable format based on ids
imdb_data = []
with open("IMDb_movies.csv") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for row in reader:
        imdb_data.append(row)

csv_columns = ['id', 'tmdb_id', 'imdb_id', 'title', 'original_title', 'adult',
               'belongs_to_collection', 'budget', 'genres', 'homepage',
               'original_language', 'overview', 'popularity', 'poster_path',
               'production_companies', 'production_countries', 'release_date',
               'revenue', 'runtime', 'spoken_languages', 'status',
               'vote_average', 'vote_count', 'year', 'metascore', 'usa_gross',
               'ww_gross', 'director', 'writer', 'production_company',
               'actor0', 'actor1', 'actor2', 'actor3', 'actor4']

# returns the queried api dictionary based on the given title
# if no information exists returns None
def title_2_api_dict (title):
    url = 'http://128.2.204.215:8080/movie/' + title[:-1]
    try:
        r = requests.get(url)
        api_info = r.json()
        imdb_id_tt = api_info['imdb_id']
        return api_info
    except (json.decoder.JSONDecodeError, KeyError):
        print("dne when queried: ", title[:-1])
        med = open("missing_external_data.txt", "r")
        med_set = set(med.readlines())
        med.close()
        if title not in med_set:
            med = open("missing_external_data.txt", "a")
            med.write(title)
            med.close()
        return None

# Returns queried api for the title plus additional data from an external imdb dataset
def title_2_advanced_api_dict (title):
    url = 'http://128.2.204.215:8080/movie/' + title[:-1]
    try:
        r = requests.get(url)
        api_info = r.json()
        imdb_id_tt = api_info['imdb_id']
    except (json.decoder.JSONDecodeError, KeyError):
        print("dne when queried or imdb missing: ", title[:-1])
        med = open("missing_external_data.txt", "r")
        med_set = set(med.readlines())
        med.close()
        if title not in med_set:
            med = open("missing_external_data.txt", "a")
            med.write(title)
            med.close()
        return None

    if imdb_id_tt in [x[0] for x in imdb_data]:  # Loop ensures existence
        for x in range(0, len(imdb_data)):
            if imdb_id_tt == imdb_data[x][0]:
                row = imdb_data[x]

                # map imdb into a dictionary
                imdb = dict(zip(header, row))

                # additional information to add from imdb data base
                api_info.update({'year': imdb['year']})
                api_info.update({'metascore': imdb['metascore']})
                api_info.update({'usa_gross': imdb['usa_gross_income']})
                api_info.update({'ww_gross': imdb['worlwide_gross_income']})
                api_info.update({'director': imdb['director']})
                api_info.update({'writer': imdb['writer']})
                api_info.update({'production_company': imdb['production_company']})
                actor_list = imdb['actors'].split(',')
                if len(actor_list) >= 5:
                    api_info.update({'actor0': actor_list[0]})
                    api_info.update({'actor1': actor_list[1]})
                    api_info.update({'actor2': actor_list[2]})
                    api_info.update({'actor3': actor_list[3]})
                    api_info.update({'actor4': actor_list[4]})
                elif len(actor_list) > 0:
                    i = 0
                    while i < len(actor_list):
                        api_info.update({('actor' + str(i)): actor_list[i]})
                        i += 1
                    while i < 5:  # backfill supporting actors with main actor
                        api_info.update({('actor' + str(i)): actor_list[0]})
                        i += 1
                else:
                    for i in range(5):
                        api_info.update({('actor' + str(i)): 'janedoe'})
        return api_info
    else:
        #print(api_info['title'] + " doesn't exist in imdb database ")
        med = open("missing_external_data.txt", "r")
        med_set = set(med.readlines())
        med.close()
        if title not in med_set:
            med = open("missing_external_data.txt", "a")
            med.write(title)
            med.close()
        return None

count = 0
for movie in fl:
    api = title_2_advanced_api_dict(movie)
    if api is None:
        continue
    #else:
    #
    if count > 10000:
        break
    else:
        count += 1

#with open('test_write.csv', mode='w') as dest:
#    writer = csv.DictWriter(dest,fieldnames=csv_columns)


#        print(api_info)

 #       f.write()
 #       break
