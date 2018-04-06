import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.system('export http_proxy=http://165.225.66.34:10015')
os.system('export https_proxy=https://165.225.66.34:10015')

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X1 = LabelEncoder()
X[:,1] = label_encoder_X1.fit_transform(X[:,1])

label_encoder_X2 = LabelEncoder()
X[:,2] = label_encoder_X2.fit_transform(X[:,2])

one_hot_encoder = OneHotEncoder(categorical_features=[1])

X = one_hot_encoder.fit_transform(X).toarray()

#To avoid dummy variable trap
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Shape of training set {}'.format(X_train.shape))

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(input_dim= 11, units= 6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units= 6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units= 1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(classifier, X_train,y_train, cv= 10)
print(accuracies)