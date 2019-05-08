import pandas as pd
import numpy as np
import keras

data = pd.read_csv('creditcard.csv')

from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'], axis = 1)
data = data.drop(['Time'], axis = 1)

X = data.drop(['Class'], axis = 1)
y = data.iloc[:, data.columns == 'Class']

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

from keras.models import Sequential
from keras.layers import Dense, Dropout 

model = Sequential([
    Dense(units=16, input_dim = 29, activation='relu'),
    Dense(units=24, activation='relu'),
    Dropout(0.5),
    Dense(20, activation='relu'),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 15, epochs = 10) 

y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)

import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
cnf_matrix = confusion_matrix(y_test, y_pred.round())    


# Undersampling
fraud_indices = np.array(data[data.Class == 1].index)
number_of_fraud_indices = len(fraud_indices)
normal_indices = data[data.Class == 0].index
random = np.random.choice(normal_indices, number_of_fraud_indices, replace = False)
random = np.array(random)

under_sample = np.concatenate([fraud_indices, random])
print(len(under_sample))

under_sample_data = data.iloc[under_sample,:]
X_under = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
y_under = under_sample_data.iloc[:,under_sample_data.columns == 'Class']

X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size = 0.2, random_state=0)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = np.array(X_train)
model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 15, epochs = 10) 


