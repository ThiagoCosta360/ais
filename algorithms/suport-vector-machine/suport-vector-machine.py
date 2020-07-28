from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys

# cancer = datasets.load_breast_cancer()
#
# print("Features: ", cancer.feature_names)
# print("Labels: ", cancer.target_names)
#
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,
#                                                     random_state=109)  # 70% training and 30% test

nRowsRead = 1000
nba_career_lenght = pd.read_csv('../datasets/nba_career_lenght/nba_longevity.csv', delimiter=',', nrows=nRowsRead,
                                header=0)
nRow, nCol = nba_career_lenght.shape
print(f'There are {nRow} rows and {nCol} columns')

features_col = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB',
                'REB', 'AST', 'STL', 'BLK', 'TOV']
X_train, X_test, y_train, y_test = train_test_split(nba_career_lenght[features_col], nba_career_lenght.TARGET_5Yrs,
                                                    test_size=0.3, random_state=120)

case = int(sys.argv[1])

if case == 0:
    svm = Pipeline([
        ("svm", LinearSVC(C=1, loss='hinge'))
    ])
elif case == 1:
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1, loss='hinge'))
    ])
elif case == 2:
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel='poly', degree=1))
    ])
elif case == 3:
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel='rbf', gamma='scale'))
    ])
# elif case == 4:
# elif case == 5:

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
