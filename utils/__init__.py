import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import time
import matplotlib.pyplot as plt

import numpy as np


def exploratory_analysis(df, target_col):
    print("\n\nDescribing pandas dataframe")
    print(df.describe().T)

    print("\n\nPrepare histograms")
    # Plot histogram
    df.hist(figsize=(15, 6))

    print("\n\nPrepare correlations")
    df_new = df.copy()
    df_new['Label'] = target_col
    corrs = df_new.corr()['Label'].sort_values(ascending=False)
    print(f"Correlations with label: {corrs}")

    print("\n\nPrepare pairplot")
    sns.pairplot(data=df_new)

    print("\n\nPrepare random forest importance")
    # Random forest importance
    feature_names = df.columns
    forest = RandomForestClassifier(random_state=0, max_depth=5, verbose=100)
    forest.fit(df, target_col)

    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def split_train_test(df, target_col, test_size=0.2):
    labels = pd.get_dummies(target_col)
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=test_size, stratify=labels)
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return scaler, X_train, X_test


def encode_labels(y_train, y_test):
    y_train_onehot = y_train.idxmax(axis=1)
    le = LabelEncoder()
    y_train_onehot = le.fit_transform(y_train_onehot)
    y_test_onehot = y_test.idxmax(axis=1)
    y_test_onehot = le.transform(y_test_onehot)

    return le, y_train_onehot, y_test_onehot


def classify_random_forest(X_train, X_test, y_train_onehot, y_test_onehot):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train_onehot)

    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train_onehot, y_pred)
    print("Train Accuracy:", accuracy)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test_onehot, y_pred)
    print("Test Accuracy:", accuracy)


def classify_svm(X_train, X_test, y_train_onehot, y_test_onehot):
    clf = SVC(gamma="auto", kernel="rbf", max_iter=1000)
    clf.fit(X_train, y_train_onehot)

    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train_onehot, y_pred)
    print("Train Accuracy:", accuracy)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test_onehot, y_pred)
    print("Test Accuracy:", accuracy)


def classify_logistic(X_train, X_test, y_train_onehot, y_test_onehot):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train_onehot)

    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train_onehot, y_pred)
    print("Train Accuracy:", accuracy)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test_onehot, y_pred)
    print("Test Accuracy:", accuracy)


def classify_GaussianNB(X_train, X_test, y_train_onehot, y_test_onehot):

    clf = GaussianNB().fit(X_train, y_train_onehot)

    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train_onehot, y_pred)
    print("Train Accuracy:", accuracy)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test_onehot, y_pred)
    print("Test Accuracy:", accuracy)

def classify_dnn(X_train, X_test, y_train, y_test, num_classes=3):
    num_inputs = X_train.shape[1]
    BATCH_SIZE = 32
    EPOCHS = 10

    classifier = tf.keras.Sequential()
    classifier.add(
        tf.keras.layers.Dense(units=10, input_dim=num_inputs, kernel_initializer='uniform', activation='relu'))
    classifier.add(tf.keras.layers.Dense(units=6, kernel_initializer='uniform', activation='relu'))

    classifier.add(tf.keras.layers.Dense(units=num_classes, kernel_initializer='uniform', activation='softmax'))

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    classifier.summary()

    history = classifier.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                             validation_data=(X_test, y_test))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

    scores = classifier.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1] * 100))


