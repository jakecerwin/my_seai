import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor, ARDRegression
from sklearn.feature_selection import chi2
from sklearn.metrics import median_absolute_error

from feature_extraction import only_numeric
from feature_extraction import advanced_clean
from feature_extraction import cut
import matplotlib
import matplotlib.pyplot as plt

basic_df = pd.read_csv('api_data.csv')
basic_df = only_numeric(basic_df)

advanced_df = pd.read_csv('advanced_api_data.csv')
advanced_df = cut(advanced_df)



data = basic_df.copy()

train1 = data.sample(frac=0.7, random_state=0)
test1 = data.drop(train1.index)

data = advanced_df.copy()

train2 = data.sample(frac=0.8, random_state=2)
test2 = data.drop(train2.index)



#data = advanced_df.copy()

#train3
#test3

#print(len(train1))
#print(len(test1))

#print(len(train2))
#print(len(test2))


# return theater popularity
def baseline1(train, test, display):

    y_pred = test['popularity']

    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(test['views'], y_pred, squared=True))
    # The coefficient of determination: 1 is perfect prediction
    print('median absolute error: %.2f'
          % median_absolute_error(test['views'], y_pred))

    plt.scatter(train['popularity'], train['views'], color='black')
    plt.plot(test['popularity'], y_pred, color='red', linewidth=1)

    plt.xticks(())
    plt.yticks(())

    # Plot outputs
    if display: plt.show()
    return y_pred

# Linear regression based solely on revenue
def baseline2(train, test, display):

    X = train.to_numpy()
    X_train = X[:, np.newaxis, train.columns.get_loc('revenue')]
    y_train = train['views']

    X = test.to_numpy()
    X_test = X[:, np.newaxis, test.columns.get_loc('revenue')]
    y_test = test['views']

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred, squared=True))
    # The coefficient of determination: 1 is perfect prediction
    print('median absolute error: %.2f'
          % median_absolute_error(y_test, y_pred))


    # Plot outputs
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='red', linewidth=1)

    plt.xticks(())
    plt.yticks(())

    if display: plt.show()
    return y_pred

def dt_regression(train, test, display):
    X = train.to_numpy()
    X_train = np.delete(X, [train.columns.get_loc('views')], axis=1)
    y_train = train['views']

    X = test.to_numpy()
    X_test = np.delete(X, [test.columns.get_loc('views')], axis=1)
    y_test = test['views']

    reg = DecisionTreeRegressor(random_state=2, max_depth=8, min_weight_fraction_leaf=.1)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    # The coefficients
    #print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred, squared=True))
    # The coefficient of determination: 1 is perfect prediction
    print('median absolute error: %.2f'
          % median_absolute_error(y_test, y_pred))

    # Plot outputs
    #plt.scatter(X_test, y_test, color='black')
    #plt.plot(X_test, y_pred, color='blue', linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #plt.show()
    return None

def ard_regression(train, test, display):
    train = train.copy()
    test = test.copy()

    X = train.to_numpy()
    X_train = np.delete(X, [train.columns.get_loc('views')], axis=1)
    y_train = train['views']

    X = test.to_numpy()
    X_test = np.delete(X, [test.columns.get_loc('views')], axis=1)
    y_test = test['views']

    reg = ARDRegression(compute_score=True)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    # The coefficients
    #print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred, squared=True))
    # The coefficient of determination: 1 is perfect prediction
    print('median absolute error: %.2f'
          % median_absolute_error(y_test, y_pred))

    # Plot outputs
    #plt.scatter(X_test, y_test, color='black')
    #plt.plot(X_test, y_pred, color='blue', linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #if display: plt.show()
    return None
#print(train)


def rf_regression(train, test, display):
    train = train.copy()
    test = test.copy()
    X = train.to_numpy()
    X_train = np.delete(X, [train.columns.get_loc('views')], axis=1)
    y_train = train['views']

    X = test.to_numpy()
    X_test = np.delete(X, [test.columns.get_loc('views')], axis=1)
    y_test = test['views']

    reg = RandomForestRegressor(random_state=0, max_depth=6)
    #reg = DecisionTreeRegressor(random_state=2, max_depth=7, min_weight_fraction_leaf=.1)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    # The coefficients
    #print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred, squared=True))
    # The coefficient of determination: 1 is perfect prediction
    print('median absolute error: %.2f'
          % median_absolute_error(y_test, y_pred))

    # Plot outputs
    #plt.scatter(X_test, y_test, color='black')
    #plt.plot(X_test, y_pred, color='blue', linewidth=3)

    #plt.xticks(())
    #plt.yticks(())

    #if display: plt.show()
    return None
#print(train)

def k_best_advanced(train, test):
    train = train.copy()
    test = test.copy()
    X = train.to_numpy()
    X_train = np.delete(X, [train.columns.get_loc('views')], axis=1)
    y_train = train['views'].astype('int')

    X = test.to_numpy()
    X_test = np.delete(X, [test.columns.get_loc('views')], axis=1)
    y_test = test['views']

    best_features = SelectKBest(score_func=chi2, k=10)
    fit = best_features.fit(X_train, y_train)
    dfscores = pd.DataFrame(fit.scores_)

    train.drop(columns='views', inplace=True)

    dfcolumns = pd.DataFrame(train.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    #print(featureScores.nlargest(20, 'Score'))  # print 10 best features

    features = ['revenue', 'budget', 'vote_count', 'runtime',
                'other_prod_co', 'in_a_collection', 'popularity',
                'metascore', 'Adventure', 'Fantasy', 'other genre',
                'release_date', 'Animation', 'Thriller', 'top100_director', 'en',
                'vote_average', 'Action', 'Science Fiction', 'homepage',
                'Documentary', 'Thriller', 'Action',
                'Fantasy']

    simple_train = train[features]

    simple_test = test[features]

    X_train = simple_train.to_numpy()
    X_test = simple_test.to_numpy()

    reg = RandomForestRegressor(random_state=0, max_depth=6)
    #reg = DecisionTreeRegressor(random_state=2, max_depth=7, min_weight_fraction_leaf=.1)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    # The coefficients
    #print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred, squared=True))
    # The coefficient of determination: 1 is perfect prediction
    print('median absolute error: %.2f'
          % median_absolute_error(y_test, y_pred))

    return None

def k_best(train, test):
    train = train.copy()
    test = test.copy()
    X = train.to_numpy()
    X_train = np.delete(X, [train.columns.get_loc('views')], axis=1)
    y_train = train['views'].astype('int')

    X = test.to_numpy()
    X_test = np.delete(X, [test.columns.get_loc('views')], axis=1)
    y_test = test['views']

    best_features = SelectKBest(score_func=chi2, k=10)
    fit = best_features.fit(X_train, y_train)
    dfscores = pd.DataFrame(fit.scores_)

    train.drop(columns='views', inplace=True)

    dfcolumns = pd.DataFrame(train.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print(featureScores.nlargest(50, 'Score'))  # print 10 best features

    features = ['revenue', 'budget', 'vote_count', 'runtime',
                'other_prod_co', 'in_a_collection',
                'Adventure', 'Fantasy', 'other genre',
                'release_date', 'Animation', 'Thriller', 'homepage', 'en',
                'vote_average', 'Action', 'Science Fiction']

    simple_train = train[features]

    simple_test = test[features]

    X_train = simple_train.to_numpy()
    X_test = simple_test.to_numpy()

    reg = RandomForestRegressor(random_state=0, max_depth=5) # 6
    #reg = DecisionTreeRegressor(random_state=2, max_depth=7, min_weight_fraction_leaf=.1)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    # The coefficients
    #print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred, squared=True))
    # The coefficient of determination: 1 is perfect prediction
    print('median absolute error: %.2f'
          % median_absolute_error(y_test, y_pred))

    return None

def plots(train):
    train.plot(x='budget', y='views', style='o')
    train.plot(x='revenue', y='views', style='o')
    train.plot(x='homepage', y='views', style='o')
    train.plot(x='release_date', y='revenue', style='o')
    train.plot(x='runtime', y='views', style='o')
    train.plot(x='status', y='views', style='o')
    train.plot(x='vote_average', y='views', style='o')
    train.plot(x='in_a_collection', y='views', style='o')
    train.plot(x='Action', y='views', style='o')
    train.plot(x='Adventure', y='views', style='o')
    train.plot(x='Animation', y='views', style='o')
    train.plot(x='other_prod_co', y='views', style='o')
    train.plot(x='summer_release', y='views', style='o')
    train.plot(x='en', y='views', style='o')
    train.plot(x='vote_count', y='views', style='o')
    train.plot(x='top100_director', y='views', style='o')
    plt.show()


print('baseline 1')
baseline1(train1, test1, False)
baseline1(train2, test2, False)
print('baseline 2')
baseline2(train1, test1, False)
baseline2(train2, test2, False)
print('dt')
dt_regression(train1, test1, False)
dt_regression(train2, test2, False)
print('k_best')
k_best(train1, test1)
k_best_advanced(train2, test2)


