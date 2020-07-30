import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


age_threshold = 25

def perf_measure(y_test, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    i = 0
    for index, test in y_test.iteritems():
        if test ==y_pred[i] == 1:
           TP += 1
        if y_pred[i]==1 and test != y_pred[i]:
           FP += 1
        if test == y_pred[i] == 2:
           TN += 1
        if y_pred[i]==2 and test != y_pred[i]:
           FN += 1
        i += 1

    return [TP, FP, TN, FN]


def data_engineering():
    german = pd.read_csv('german.data', sep=" ")

    profiles = list()
    # Feature extraction and encoding
    for index, row in german.iterrows():
        user = dict()

        status = row['account_status']
        if status == 'A11':
            status = 0
        elif status == 'A12':
            status = 1
        elif status == 'A13':
            status = 2
        else:
            status = 3
        user.update({'account_status': status})

        user.update({'duration': row['duration']})

        status = row['credit_history']
        if status == 'A31':
            status = 1
        elif status == 'A32':
            status = 2
        elif status == 'A30':
            status = 0
        elif status == 'A34':
            status = 4
        else:
            status = 3
        user.update({'credit_history': status})

        status = row['purpose']
        if status == 'A40':
            status = 0
        elif status == 'A41':
            status = 1
        elif status == 'A42':
            status = 2
        elif status == 'A43':
            status = 3
        elif status == 'A44':
            status = 4
        elif status == 'A45':
            status = 5
        elif status == 'A46':
            status = 6
        elif status == 'A47':
            status = 7
        elif status == 'A48':
            status = 8
        elif status == 'A49':
            status = 9
        else:
            status = 10
        user.update({'purpose': status})

        user.update({'credit_amount': row['credit_amount']})

        status = row['savings_account']
        if status == 'A65':
            status = 0
        elif status == 'A61':
            status = 1
        elif status == 'A62':
            status = 2
        elif status == 'A63':
            status = 3
        else:
            status = 4
        user.update({'savings_account': status})

        status = row['employed_since']
        if status == 'A71':
            status = 0
        elif status == 'A72':
            status = 1
        elif status == 'A73':
            status = 2
        elif status == 'A74':
            status = 3
        else:
            status = 4
        user.update({'employed_since': status})

        user.update({'installment_rate': row['installment_rate']})

        # sex : 1 = male, 0 = female
        status = row['status_and_sex']
        if status == 'A91':
            sex = 1
            status = 1
        elif status == 'A92':
            sex = 0
            status = 0.5
        elif status == 'A93':
            sex = 1
            status = 1
        elif status == 'A94':
            sex = 1
            status = 0
        else:
            sex = 0
            status = 1
        user.update({'sex': sex})
        user.update({'relationship_status': status})

        status = row['guarantors']
        if status == 'A101':
            status = 1
        elif status == 'A102':
            status = 2
        else:
            status = 3
        user.update({'guarantors': status})

        user.update({'residence_since': row['residence_since']})

        status = row['property']
        if status == 'A121':
            status = 1
        elif status == 'A122':
            status = 2
        elif status == 'A123':
            status = 3
        else:
            status = 4
        user.update({'property': status})

        user.update({'age': row['age']})

        status = row['other_installment']
        if status == 'A141':
            status = 1
        elif status == 'A142':
            status = 2
        else:
            status = 3
        user.update({'other_installment': status})

        status = row['housing']
        if status == 'A141':
            status = 1
        elif status == 'A142':
            status = 2
        else:
            status = 3
        user.update({'housing': status})

        user.update({'num_existing_credits': row['num_existing_credits']})

        status = row['job']
        if status == 'A171':
            status = 1
        elif status == 'A172':
            status = 2
        elif status == 'A173':
            status = 3
        else:
            status = 4
        user.update({'job': status})

        user.update({'liable_people': row['liable_people']})

        status = row['telephone']
        if status == 'A191':
            status = 1
        else:
            status = 2
        user.update({'telephone': status})

        status = row['foreign']
        if status == 'A201':
            status = 1
        else:
            status = 0
        user.update({'foreign': status})

        user.update({'good_bad': row['good_bad']})

        profiles.append(user)

    german = pd.DataFrame(profiles)

    german.plot.box(y='age', x='good_bad')

    n, m = german.shape

    X = german.iloc[:, 0:m - 1]
    y = german.iloc[:, m - 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def measuring_fairness(model, X_test, y_test):

    y_pred = model.predict(X_test)

    gender_inverted = X_test.copy()
    age_inverted = X_test.copy()
    gender_inverted['sex'] = gender_inverted['sex'].apply(lambda x: int(x == 0))
    age_inverted['age'] = age_inverted['age'].apply(lambda x: int(x <= age_threshold) * 25)
    y_gender_inverted_pred = model.predict(gender_inverted)
    y_age_inverted_pred = model.predict(age_inverted)

    y_pred_series = pd.Series(y_pred, index=y_test.index.values)

    y_pred_df = pd.DataFrame(y_pred_series, columns=['pred'])

    good = X_test[y_pred_df['pred'] == 1]

    good_male = good[good['sex'] == 1]
    good_female = good[good['sex'] != 1]

    male = X_test[X_test['sex'] == 1]
    female = X_test[X_test['sex'] != 1]

    good_older = good[good['age'] > age_threshold]
    good_young = good[good['age'] <= age_threshold]

    older = X_test[X_test['age'] > age_threshold]
    young = X_test[X_test['age'] <= age_threshold]

    male_TP, male_FP, male_TN, male_FN = perf_measure(y_test[X_test['sex'] == 1], y_pred[X_test['sex'] == 1])
    female_TP, female_FP, female_TN, female_FN = perf_measure(y_test[X_test['sex'] == 0], y_pred[X_test['sex'] == 0])
    young_TP, young_FP, young_TN, young_FN = perf_measure(y_test[X_test['age'] <= age_threshold],
                                                          y_pred[X_test['age'] <= age_threshold])
    older_TP, older_FP, older_TN, older_FN = perf_measure(y_test[X_test['age'] > age_threshold],
                                                          y_pred[X_test['age'] > age_threshold])

    print("roc auc :", metrics.roc_auc_score(y_test, y_pred))
    print("Anti-Classification: ")
    print("uninverted Recall     :", metrics.recall_score(y_test, y_pred))
    print("inverted gender Recall:", metrics.recall_score(y_test, y_gender_inverted_pred))
    print("inverted age Recall   :", metrics.recall_score(y_test, y_age_inverted_pred))
    print()
    print("Indepedence: ")
    print("P(Good | Male) :", len(good_male) / len(male))
    print("P(Good | Female) :", len(good_female) / len(female))
    print("Independence achieved within 2%:",
          .02 > abs((len(good_male) / len(male)) - (len(good_female) / len(female))))
    print("P(Good | older) :", len(good_older) / len(older))
    print("P(Good | young) :", len(good_young) / len(young))
    print("Independence achieved within 2%:",
          .02 > abs((len(good_older) / len(older)) - (len(good_young) / len(young))))
    print()
    print("Seperation as binary target rate:")
    print("Male True Positive Rate:", (male_TP / (male_TP + male_FN)),
          "False Positive Rate:", (male_FP / (male_TN + male_FP)))
    print("Female True Positive Rate:", (female_TP / (female_TP + female_FN)),
          "False Positive Rate:", (female_FP / (female_TN + female_FP)))
    print("young True Positive Rate", (young_TP / (young_TP + young_FN)),
          "False Positive Rate:", (young_FP / (young_TN + young_FP)))
    print("older True Positive Rate", (older_TP / (older_TP + older_FN)),
          "False Positive Rate:", (older_FP / (older_TN + older_FP)))
    print()

    return None


def train_anti_classification_model(X_train, y_train):

    X_train = X_train.copy()
    X_train.drop(labels=['sex', 'age'], inplace=True, axis=1)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def measuring_fairness_anti_classification_model(model, X_test, y_test):
    X_test_filter = X_test.copy()
    X_test_filter.drop(labels=['sex', 'age'], inplace=True, axis=1)

    y_pred = model.predict(X_test_filter)

    X_male_test = X_test[X_test['sex'] == 1].drop(labels=['sex', 'age'], axis=1)
    y_male_pred = model.predict(X_male_test)
    y_male_test = y_test[X_test['sex'] == 1]

    X_female_test = X_test[X_test['sex'] != 1].drop(labels=['sex', 'age'], axis=1)
    y_female_pred = model.predict(X_female_test)
    y_female_test = y_test[X_test['sex'] != 1]

    X_young_test = X_test[X_test['age'] <= age_threshold].drop(labels=['sex', 'age'], axis=1)
    y_young_pred = model.predict(X_young_test)
    y_young_test = y_test[X_test['age'] <= age_threshold]

    X_older_test = X_test[X_test['age'] > age_threshold].drop(labels=['sex', 'age'], axis=1)
    y_older_pred = model.predict(X_older_test)
    y_older_test = y_test[X_test['age'] > age_threshold]

    male_recall = metrics.recall_score(y_male_test, y_male_pred)
    female_recall = metrics.recall_score(y_female_test, y_female_pred)
    young_recall = metrics.recall_score(y_young_test, y_young_pred)
    older_recall = metrics.recall_score(y_older_test, y_older_pred)

    sex_equality = female_recall - male_recall
    age_equality = young_recall - older_recall

    y_pred_series = pd.Series(y_pred, index=y_test.index.values)

    y_pred_df = pd.DataFrame(y_pred_series, columns=['pred'])

    good = X_test[y_pred_df['pred'] == 1]

    good_male = good[good['sex'] == 1]
    good_female = good[good['sex'] != 1]

    male = X_test[X_test['sex'] == 1]
    female = X_test[X_test['sex'] != 1]

    good_older = good[good['age'] > age_threshold]
    good_young = good[good['age'] <= age_threshold]

    older = X_test[X_test['age'] > age_threshold]
    young = X_test[X_test['age'] <= age_threshold]

    male_TP, male_FP, male_TN, male_FN = perf_measure(y_test[X_test['sex'] == 1], y_pred[X_test['sex'] == 1])
    female_TP, female_FP, female_TN, female_FN = perf_measure(y_test[X_test['sex'] == 0], y_pred[X_test['sex'] == 0])
    young_TP, young_FP, young_TN, young_FN = perf_measure(y_test[X_test['age'] <= age_threshold],
                                                          y_pred[X_test['age'] <= age_threshold])
    older_TP, older_FP, older_TN, older_FN = perf_measure(y_test[X_test['age'] > age_threshold],
                                                          y_pred[X_test['age'] > age_threshold])

    print("Fairness measures taken")
    print("roc auc :", metrics.roc_auc_score(y_test, y_pred))
    print("Anti-Classification: ")
    print("Recall     :", metrics.recall_score(y_test, y_pred))
    print()
    print("Indepedence: ")
    print("P(Good | Male) :", len(good_male) / len(male))
    print("P(Good | Female) :", len(good_female) / len(female))
    print("Independence achieved within 2%:",
          .02 > abs((len(good_male) / len(male)) - (len(good_female) / len(female))))
    print("P(Good | older) :", len(good_older) / len(older))
    print("P(Good | young) :", len(good_young) / len(young))
    print("Independence achieved within 2%:",
          .02 > abs((len(good_older) / len(older)) - (len(good_young) / len(young))))
    print()
    print("Seperation as binary target rate:")
    print("Male True Positive Rate:", (male_TP / (male_TP + male_FN)),
          "False Positive Rate:", (male_FP / (male_TN + male_FP)))
    print("Female True Positive Rate:", (female_TP / (female_TP + female_FN)),
          "False Positive Rate:", (female_FP / (female_TN + female_FP)))
    print("young True Positive Rate", (young_TP / (young_TP + young_FN)),
          "False Positive Rate:", (young_FP / (young_TN + young_FP)))
    print("older True Positive Rate", (older_TP / (older_TP + older_FN)),
          "False Positive Rate:", (older_FP / (older_TN + older_FP)))

    return None


def train_indepedent_model(X_train, y_train):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.60, random_state=21)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model = model.fit(X_train, y_train)

    X_male = X_test[X_test['sex'] == 1]
    y_male = y_test[X_test['sex'] == 1]
    X_female = X_test[X_test['sex'] != 1]
    y_female = y_test[X_test['sex'] != 1]

    male_pred = model.predict(X_male)
    female_pred = model.predict(X_female)

    male_cutoff = 1.5
    female_cutoff = 1.535

    male_pred_series = pd.Series(male_pred, index=y_male.index.values)
    female_pred_series = pd.Series(female_pred, index=y_female.index.values)

    male_pred_df = pd.DataFrame(male_pred_series, columns=['pred'])
    female_pred_df = pd.DataFrame(female_pred_series, columns=['pred'])

    pmale_pred_df = pd.DataFrame(male_pred_series, columns=['pred'])
    pfemale_pred_df = pd.DataFrame(female_pred_series, columns=['pred'])

    pmale_pred_df['pred'] = pmale_pred_df['pred'].apply(lambda x: (int(x >= male_cutoff) + 1))
    pfemale_pred_df['pred'] = pfemale_pred_df['pred'].apply(lambda x: (int(x >= male_cutoff) + 1))

    pgood_male = X_male[pmale_pred_df['pred'] == 1]
    pgood_female = X_female[pfemale_pred_df['pred'] == 1]

    pmale = len(pgood_male)
    pfemale = len(pgood_female)


    male_pred_df['pred'] = male_pred_df['pred'].apply(lambda x: (int(x >= male_cutoff) + 1))
    female_pred_df['pred'] = female_pred_df['pred'].apply(lambda x: (int(x >= female_cutoff) + 1))

    good_male = X_male[male_pred_df['pred'] == 1]
    good_female = X_female[female_pred_df['pred'] == 1]

    male = len(good_male)
    female = len(good_female)

    print()
    print("Raw Recall independence: ")
    print("P(Good | Male) :", pmale / len(X_male))
    print("P(Good | Female) :", pfemale / len(X_female))
    print()
    print("Recall independence after threshold adjustment: ")
    print("P(Good | Male) :", male / len(X_male))
    print("P(Good | Female) :", female / len(X_female))


    return [model, male_cutoff, female_cutoff]


def measuring_fairness_independent_model(model, X_test, y_test, male_cutoff, female_cutoff):

    X_male = X_test[X_test['sex'] == 1]
    y_male = y_test[X_test['sex'] == 1]
    X_female = X_test[X_test['sex'] != 1]
    y_female = y_test[X_test['sex'] != 1]

    male_pred = model.predict(X_male)
    female_pred = model.predict(X_female)

    male_pred_series = pd.Series(male_pred, index=y_male.index.values)
    female_pred_series = pd.Series(female_pred, index=y_female.index.values)

    male_pred_df = pd.DataFrame(male_pred_series, columns=['pred'])
    female_pred_df = pd.DataFrame(female_pred_series, columns=['pred'])

    male_pred_df['pred'] = male_pred_df['pred'].apply(lambda x: (int(x >= male_cutoff) + 1))
    female_pred_df['pred'] = female_pred_df['pred'].apply(lambda x: (int(x >= female_cutoff) + 1))

    y_pred = male_pred_df.append(female_pred_df).sort_index()
    y_test = y_test.sort_index()

    good_male = X_male[male_pred_df['pred'] == 1]
    good_female = X_female[female_pred_df['pred'] == 1]

    male = len(good_male)
    female = len(good_female)

    print()
    print("Recall independence on validation data: ")
    print("P(Good | Male) :", male / len(X_male))
    print("P(Good | Female) :", female / len(X_female))
    print("roc auc :", metrics.roc_auc_score(y_test, y_pred))

    return None


def train_separate_model(X_train, y_train):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.60, random_state=21)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model = model.fit(X_train, y_train)

    X_male = X_test[X_test['sex'] == 1]
    y_male = y_test[X_test['sex'] == 1]
    X_female = X_test[X_test['sex'] != 1]
    y_female = y_test[X_test['sex'] != 1]

    male_pred = model.predict(X_male)
    female_pred = model.predict(X_female)

    male_cutoff = 1.5
    female_cutoff = 1.535

    male_pred_series = pd.Series(male_pred, index=y_male.index.values)
    female_pred_series = pd.Series(female_pred, index=y_female.index.values)

    male_pred_df = pd.DataFrame(male_pred_series, columns=['pred'])
    female_pred_df = pd.DataFrame(female_pred_series, columns=['pred'])

    pmale_pred_df = pd.DataFrame(male_pred_series, columns=['pred'])
    pfemale_pred_df = pd.DataFrame(female_pred_series, columns=['pred'])

    pmale_pred_df['pred'] = pmale_pred_df['pred'].apply(lambda x: (int(x >= male_cutoff) + 1))
    pfemale_pred_df['pred'] = pfemale_pred_df['pred'].apply(lambda x: (int(x >= male_cutoff) + 1))

    pgood_male = X_male[pmale_pred_df['pred'] == 1]
    pgood_female = X_female[pfemale_pred_df['pred'] == 1]

    pmale = len(pgood_male)
    pfemale = len(pgood_female)


    male_pred_df['pred'] = male_pred_df['pred'].apply(lambda x: (int(x >= male_cutoff) + 1))
    female_pred_df['pred'] = female_pred_df['pred'].apply(lambda x: (int(x >= female_cutoff) + 1))

    male_TP, male_FP, male_TN, male_FN = perf_measure(y_male[X_test['sex'] == 1], pmale_pred_df.to_numpy())
    female_TP, female_FP, female_TN, female_FN = perf_measure(y_female[X_test['sex'] == 0], pfemale_pred_df.to_numpy())

    print()
    print("Raw independence")
    print("Male True Positive Rate:", (male_TP / (male_TP + male_FN)),
          "False Positive Rate:", (male_FP / (male_TN + male_FP)))
    print("Female True Positive Rate:", (female_TP / (female_TP + female_FN)),
          "False Positive Rate:", (female_FP / (female_TN + female_FP)))

    male_TP, male_FP, male_TN, male_FN = perf_measure(y_male[X_test['sex'] == 1], male_pred_df.to_numpy())
    female_TP, female_FP, female_TN, female_FN = perf_measure(y_female[X_test['sex'] == 0], female_pred_df.to_numpy())

    print()
    print("Independence after threshold adjustment:")
    print("Male True Positive Rate:", (male_TP / (male_TP + male_FN)),
          "False Positive Rate:", (male_FP / (male_TN + male_FP)))
    print("Female True Positive Rate:", (female_TP / (female_TP + female_FN)),
          "False Positive Rate:", (female_FP / (female_TN + female_FP)))


    return [model, male_cutoff, female_cutoff]


def measuring_fairness_seperate_model(model, X_test, y_test, male_cutoff, female_cutoff):

    X_male = X_test[X_test['sex'] == 1]
    y_male = y_test[X_test['sex'] == 1]
    X_female = X_test[X_test['sex'] != 1]
    y_female = y_test[X_test['sex'] != 1]

    male_pred = model.predict(X_male)
    female_pred = model.predict(X_female)

    male_pred_series = pd.Series(male_pred, index=y_male.index.values)
    female_pred_series = pd.Series(female_pred, index=y_female.index.values)

    male_pred_df = pd.DataFrame(male_pred_series, columns=['pred'])
    female_pred_df = pd.DataFrame(female_pred_series, columns=['pred'])

    male_pred_df['pred'] = male_pred_df['pred'].apply(lambda x: (int(x >= male_cutoff) + 1))
    female_pred_df['pred'] = female_pred_df['pred'].apply(lambda x: (int(x >= female_cutoff) + 1))

    male_TP, male_FP, male_TN, male_FN = perf_measure(y_male[X_test['sex'] == 1], male_pred_df.to_numpy())
    female_TP, female_FP, female_TN, female_FN = perf_measure(y_female[X_test['sex'] == 0], female_pred_df.to_numpy())

    print()
    print("Independence on validation data")
    print("Male True Positive Rate:", (male_TP / (male_TP + male_FN)),
          "False Positive Rate:", (male_FP / (male_TN + male_FP)))
    print("Female True Positive Rate:", (female_TP / (female_TP + female_FN)),
          "False Positive Rate:", (female_FP / (female_TN + female_FP)))
    y_pred = male_pred_df.append(female_pred_df).sort_index()
    y_test = y_test.sort_index()
    print("roc auc :", metrics.roc_auc_score(y_test, y_pred))

    return None



X_train, X_test, y_train, y_test = data_engineering()

model = train_model(X_train, y_train)
metrics.plot_roc_curve(X=X_test, y=y_test, estimator=model)

print("No fairness measures taken")
measuring_fairness(model, X_test, y_test)

model = train_anti_classification_model(X_train, y_train)
measuring_fairness_anti_classification_model(model, X_test, y_test)

print("more independence")
model, male_cutoff, female_cutoff = train_indepedent_model(X_train, y_train)
measuring_fairness_independent_model(model, X_test, y_test, male_cutoff, female_cutoff)

print("seperation")
model, male_cutoff, female_cutoff = train_seperate_model(X_train, y_train)
measuring_fairness_seperate_model(model, X_test, y_test, male_cutoff, female_cutoff)




