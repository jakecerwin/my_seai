import csv

# drops title columns

views = dict()
with open("views.csv") as csv_views:
    reader = csv.reader(csv_views)
    for row in reader:
        views.update({row[0]: row[1]})


def simple_clean(df):
    to_drop = ['Unnamed: 0', 'id', 'tmdb_id', 'imdb_id', 'title',
               'original_title'],
    for item in to_drop:
        df.drop(columns=item, inplace=True)
    return df


# converts df into all numeric values
def only_numeric(df):
    def try_read(x):
        try:
            return views[x]
        except KeyError:
            return 0

    df['views'] = df['id'].apply(lambda x: try_read(x))

    df = simple_clean(df)
    nan_removed = df.notnull().astype('int')
    df['homepage'] = nan_removed['homepage']

    df = df.dropna()

    # transfom release_date into release year only
    df['summer_release'] = df['release_date'].apply(lambda x: float(4 < int(x[5:7]) < 9))
    #print(type(df['release_date']))
    df['release_date'] = df['release_date'].apply(lambda x: int(x[0:4]) + (int(x[5:7]) / 12))

    df['adult'] = df['adult'].apply(lambda x: float(int(x)))
    df['status'] = df['status'].apply(lambda x: float(int(x == 'Released')))
    df['original_english'] = df['original_language'].apply(lambda x: float(int(x == 'en')))
    df.drop(columns='original_language', inplace=True)

    for item in df.keys():
        #print(df[item])
        df[item] = df[item].apply(lambda x: float(x))

    return df


def advanced_clean(df):

    #def fill_gross(x):


    nan_removed = df.notnull().astype('int')
    df['homepage'] = nan_removed['homepage']

    df['metascore'].fillna((df['metascore'].mean()), inplace=True)

    df.drop(columns='ww_gross', inplace=True)
    df.drop(columns='usa_gross', inplace=True)

    #df['ww_gross'] = df['ww_gross'].fillna('$ '+ df['revenue'])

    df = df.dropna()
    df['top100_director'] = df['top100_director'].apply(lambda x: float(x))
    #df['ww_gross'] = df['ww_gross'].apply(lambda x: x[2:])
    #df['usa_gross'] = df['usa_gross'].apply(lambda x: x[2:])
    df = only_numeric(df)

    # only have to clean usa_gross, ww_gross, top100_director


    return df


def cut(df):
    major_prod_cos =['New Line Cinema', 'Twentieth Century Fox Film Corporation',
                     'Miramax Films', 'TriStar Pictures', 'United Artists',
                     'Paramount Pictures', 'Columbia Pictures',
                     'Walt Disney Pictures', 'Warner Bros.',
                     'Metro-Goldwyn-Mayer (MGM)', 'Universal Pictures',
                     'Columbia Pictures Corporation', 'Touchstone Pictures',
                     'Canal+']

    df.drop(columns=major_prod_cos, inplace=True)

    #print(df.keys)
    return advanced_clean(df)


