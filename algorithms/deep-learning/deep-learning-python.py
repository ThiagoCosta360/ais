# Import pandas
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

white = pd.read_csv(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')


# Add `type` column to `white` with value 0
white['type'] = 0
# Add `type` column to `red` with value 1
red['type'] = 1
# Append `white` to `red`
wines = red.append(white, ignore_index=True)

# Specify the data
X = wines.iloc[:, 0:11]
# Specify the target labels and flatten the array
Y = np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

# Define the scaler
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)

# Initialize the constructor
model = Sequential()
# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(11,)))
# Add one hidden layer
model.add(Dense(8, activation='relu'))
# Add an output layer
model.add(Dense(1, activation='sigmoid'))

# Model output shape
model.output_shape

# Model summary
model.summary()
# Model config
model.get_config()
# List all weight tensors
model.get_weights()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=4352, verbose=1)

y_pred = model.predict(X_test).round()

y_pred[:5]
y_test[:5]

score = model.evaluate(X_test, y_test, verbose=1)

print(score)

print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(cohen_kappa_score(y_test, y_pred))
