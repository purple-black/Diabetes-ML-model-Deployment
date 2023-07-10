import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn

def remove_outliers_iqr(data, column, threshold=1.5):
    q25 = np.percentile(data[column], 25)
    q75 = np.percentile(data[column], 75)
    iqr = q75 - q25
    lower_bound = q25 - threshold * iqr
    upper_bound = q75 + threshold * iqr
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

df = pd.read_csv('diabetes.csv')
print(df.head())
#remove outliers
diabetes_filtered = remove_outliers_iqr(df, 'Glucose')

from sklearn.model_selection import train_test_split 
X = df.iloc[:, 0:8]  # Assigning the first 8 columns of the DataFrame to X as the features
Y = df.iloc[:, 8]  # Assigning the last column of the DataFrame to Y as the target variable
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  # Splitting the data into training and testing sets
X_train

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # Creating an instance of the StandardScaler class
X_train = scaler.fit_transform(X_train)  # Scaling the training set features using the fit_transform()
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, Y_train)
ypred = classifier.predict(X_test)
print(classification_report(Y_test,ypred))
print(confusion_matrix(Y_test, ypred))

#pickle
pickle.dump(classifier, open("model.pkl", "wb"))