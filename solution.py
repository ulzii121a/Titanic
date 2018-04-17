import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def init():
    # Reads Dataset:
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    combine = [train_df, test_df]

    analyzeByClass = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False)

    # Drops Ticket and Cabin:
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    # Creates Title:
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    print(train_df.head())

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    print(train_df.shape, test_df.shape)

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    guess_ages = np.zeros((2, 3))

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) &
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                age_guess = guess_df.median()
                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1),
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
                                                                                                    ascending=True))

    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]
    print(train_df.head())

    # Creates new Feature:
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                          ascending=False))
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    freq_port = train_df.Embarked.dropna().mode()[0]

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                      ascending=False))
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    print(test_df.head())

    # Creates Fere Band:
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                                      ascending=True))

    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

    print(train_df.head(10))
    print(test_df.head(10))

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()
    print(X_train.shape, Y_train.shape, X_test.shape)

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print(acc_log)

    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
    print(coeff_df.sort_values(by='Correlation', ascending=False))

    # Support Vector Machines
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    print(acc_svc)

    # KNN:
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    print(acc_knn)

    # Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    print(acc_gaussian)

    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    print(acc_decision_tree)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)

    models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                  'Random Forest', 'Naive Baise', 'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log,
                  acc_random_forest, acc_gaussian, acc_decision_tree]})
    print(models.sort_values(by='Score', ascending=False))

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

    submission.to_csv('output/submission.csv', index=False)


def main():
    init()


main()
