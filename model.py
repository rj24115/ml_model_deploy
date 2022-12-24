import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import pickle

# Load the csv file
df = pd.read_csv("iris.csv")

print(df.head())

# Select independent and dependent variable
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

# Encode for string labels
# label_encoder = LabelEncoder().fit(y)
# y = label_encoder.transform(y)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()
clf2 = LogisticRegression(random_state=0)
clf3 = GradientBoostingClassifier()

# Fit the model
classifier.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))
pickle.dump(clf2, open("model2.pkl", "wb"))
pickle.dump(clf3, open("model3.pkl", "wb"))

print('RandomForest score: ', classifier.score(X_test, y_test))
print('Logistic Regression score: ', clf2.score(X_test, y_test))
print('Gradient Boosting score: ', clf3.score(X_test, y_test))

# multiclass
# print('RandomForest auc roc score: ', roc_auc_score(y_test, classifier.predict(X_test)) )
# print('Logistic Regression auc roc score: ', roc_auc_score(y_test, clf2.predict(X_test)))