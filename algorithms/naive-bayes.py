from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny',
           'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
        'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']

play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
        'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

le = preprocessing.LabelEncoder()
weather_encoded = le.fit_transform(weather)
temp_encoded = le.fit_transform(temp)
label = le.fit_transform(play)

features = list(zip(weather_encoded, temp_encoded))

model = GaussianNB()

model.fit(features, label)

predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)

# wine = datasets.load_wine()
# X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)
#
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred = gnb.predict(X_test)
#
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

nRowsRead = 1000
nba_career_lenght = pd.read_csv('../datasets/nba_career_lenght/nba_longevity.csv', delimiter=',', nrows=nRowsRead,
                                header=0)
nRow, nCol = nba_career_lenght.shape
print(f'There are {nRow} rows and {nCol} columns')

# features_col = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB',
#                 'REB', 'AST', 'STL', 'BLK', 'TOV']
features_col = ['GP', 'MIN', 'PTS', 'FG%', '3P%',
                'FT%', 'REB', 'AST', 'STL', 'BLK', 'TOV']
X_train, X_test, y_train, y_test = train_test_split(nba_career_lenght[features_col], nba_career_lenght.TARGET_5Yrs,
                                                    test_size=0.3, random_state=120)

nb = GaussianNB(var_smoothing=1e-1)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
