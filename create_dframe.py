import json
import requests
import csv
import pandas as pd


# Transform additional csv dataframe(s) into searchable format based on ids
imdb_data = []
with open("IMDb_movies.csv") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for row in reader:
        imdb_data.append(row)

csv_columns = ['id', 'tmdb_id', 'imdb_id', 'title', 'original_title', 'adult',
               'belongs_to_collitaection', 'budget', 'genres', 'homepage',
               'original_language', 'overview', 'popularity', 'poster_path',
               'production_companies', 'production_countries', 'release_date',
               'revenue', 'runtime', 'spoken_languages', 'status',
               'vote_average', 'vote_count', 'year', 'metascore', 'usa_gross',
               'ww_gross', 'director', 'writer', 'production_company',
               'actor0', 'actor1', 'actor2', 'actor3', 'actor4']


# returns the queried api json dictionary based on the given title
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

# Returns queried api json dict for the title plus additional data from an external imdb dataset
def title_2_advanced_api_dict (title):
    url = 'http://128.2.204.215:8080/movie/' + title[:-1]
    try:
        r = requests.get(url)
        api_info = r.json()
        imdb_id_tt = api_info['imdb_id']
    except (json.decoder.JSONDecodeError, KeyError):
        print("dne when queried or imdb missing: ", title[:-1])
        return None



    if imdb_id_tt in [x[0] for x in imdb_data]:  # Loop ensures existence
        for x in range(0, len(imdb_data)):
            if imdb_id_tt == imdb_data[x][0]:
                row = imdb_data[x]

                # map imdb into a dictionary
                imdb = dict(zip(header, row))

                # list of top 100 actors according to imdb
                top100 = set([
                    'Stanley Kubrick', 'Ingmar Bergman', 'Alfred Hitchcock',
                    'Akira Kurosawa', 'Orson Welles', 'Federico Fellini',
                    'John Ford', 'Jean-Luc Godard', 'Luis Buñuel',
                    'Martin Scorsese', 'Robert Bresson', 'Charles Chaplin',
                    'Jean Renoir', 'Howard Hawks', 'Steven Spielberg',
                    'Michelangelo Antonioni', 'Andrei Tarkovsky', 'David Lynch',
                    'Yasujirô Ozu', 'Billy Wilder', 'Fritz Lang',
                    'Carl Theodor Dreyer', 'Francis Ford Coppola', 'F.W. Murnau',
                    'Terrence Malick', 'Sergei M. Eisenstein', 'David Lean',
                    'Michael Powell', 'François Truffaut', 'Kenji Mizoguchi',
                    'Woody Allen', 'Robert Altman', 'Vittorio De Sica',
                    'Satyajit Ray', 'Sidney Lumet', 'Roman Polanski',
                    'Roberto Rossellini', 'Luchino Visconti', 'John Cassavetes',
                    'Sergio Leone', ' D.W. Griffith', ' Buster Keaton',
                    'Werner Herzog', 'Krzysztof Kieslowski', 'Abbas Kiarostami',
                    'Béla Tarr', 'Michael Haneke', 'Lars von Trier', 'Joel Coen',
                    'Joel Coen, Ethan Coen', 'Ethan Coen, Joel Coen',
                    'Quentin Tarantino', 'John Huston', 'Frank Capra',
                    'Pedro Almodóvar', 'Kar-Wai Wong', 'David Fincher',
                    'Jean-Pierre Melville', 'Henri-Georges Clouzot',
                    'William Wyler', ' Elia Kazan', 'Christopher Nolan',
                    'Richard Linklater', 'Mike Leigh', 'Yimou Zhang',
                    'Spike Lee', 'Douglas Sirk', 'Alain Resnais',
                    'Jacques Tati', 'Oliver Stone', 'Brian De Palma',
                    'Rainer Werner Fassbinder', 'Wim Wenders', 'Hsiao-Hsien Hou',
                    'David Cronenberg', 'Edward Yang', 'Terry Gilliam',
                    'Pier Paolo Pasolini', 'Bernardo Bertolucci', 'Ridley Scott',
                    'James Cameron', 'Max Ophüls', 'Ernst Lubitsch',
                    'Josef von Sternberg', 'Jacques Demy', 'Preston Sturges',
                    'Jean Cocteau', 'Mike Nichols', 'Milos Forman',
                    'Alfonso Cuarón', 'Alejandro G. Iñárritu', 'Hayao Miyazaki',
                    'Sam Peckinpah', 'Samuel Fuller', 'Chantal Akerman',
                    'Agnès Varda', 'Nicolas Roeg', 'Ken Loach', 'Wes Anderson',
                    'Darren Aronofsky', 'Alejandro Jodorowsky'])

                # additional information to add from imdb data base
                #api_info.update({'year': imdb['year']}) redundant
                api_info.update({'metascore': imdb['metascore']})
                api_info.update({'usa_gross': imdb['usa_gross_income']})
                api_info.update({'ww_gross': imdb['worlwide_gross_income']})
                if imdb['director'] in top100:
                    api_info.update({'top100_director': True})
                else:
                    api_info.update({'top100_director': False})

                #api_info.update({'writer': imdb['writer']})
                #api_info.update({'production_company': imdb['production_company']})
                #actor_list = imdb['actors'].split(',')
                #if len(actor_list) >= 5:
                #    api_info.update({'actor0': actor_list[0]})
                #    api_info.update({'actor1': actor_list[1]})
                #    api_info.update({'actor2': actor_list[2]})
                #    api_info.update({'actor3': actor_list[3]})
                #    api_info.update({'actor4': actor_list[4]})
                #elif len(actor_list) > 0:
                #    i = 0
                #    while i < len(actor_list):
                #        api_info.update({('actor' + str(i)): actor_list[i]})
                #        i += 1
                #    while i < 5:  # backfill supporting actors with main actor
                #        api_info.update({('actor' + str(i)): actor_list[0]})
                #        i += 1
                #else:
                #    for i in range(5):
                #        api_info.update({('actor' + str(i)): 'janedoe'})
        return api_info
    else:
        return None


# Given .txt file containing tiles in name+name+year form
# will construct a dataframe with queried data
def create_basic_dframe(src):
    f = open(src, "r")
    fl = f.readlines()
    f.close()
    data = list()
    i = 0
    for movie in fl:
        #if i > 100: break
        api = title_2_api_dict(movie)
        if api is None:
            continue
        else:
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
                    api.update({name: (count+1)})
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
                    api.update({name: (count+1)})
                else:
                    count = api['other_prod_co']
                    api.update({'other_prod_co': count + 1})

            del api['production_companies']

            # convert spoken languages into one hot encoding
            spoken_languages = api['spoken_languages']

            #most common production companies
            api.update({'en': 0})
            api.update({'es': 0})
            api.update({'fr': 0})
            api.update({'it': 0})
            api.update({'de': 0})
            api.update({'other_spk_lng': 0})
            lgs = ['en','es','fr','it','de']

            for language in spoken_languages:
                name = language['iso_639_1']
                if name in lgs:
                    count = api[name]
                    api.update({name: count+1})
                else:
                    count = api['other_spk_lng']
                    api.update({'other_spk_lng': count + 1})

            del api['spoken_languages']

            del api['production_countries']

            data.append(api)
            i += 1
    return pd.DataFrame(data)

# Given .txt file containing tiles in name+name+year form
# will construct a pandas dataframe with queried data and additional imdb dat
def create_advanced_dframe(src):
    f = open(src, "r")
    fl = f.readlines()
    f.close()
    data = list()
    i = 0
    for movie in fl:
        #if i > 1000: break
        api = title_2_advanced_api_dict(movie)
        if api is None:
            continue
        else:
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

            data.append(api)
            i += 1
    return pd.DataFrame(data)

# Given .txt file containing tiles in name+name+year form
# will construct an api csv file with queried data
# and save it to dest
def save_api(src, dest):
    df = create_basic_dframe(src)
    df.to_csv(dest)

# Given .txt file containing tiles in name+name+year form
# will construct an api csv file with queried data and additional imdb data
def save_advanced_api(src, dest):
    df = create_advanced_dframe(src)
    df.to_csv(dest)


save_advanced_api('movie_titles.txt', 'advanced_api_data.csv')
save_api('movie_titles.txt', 'api_data.csv')
