from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import model_selection
import numpy as np
import pickle

import pandas as pd


def train_model(data_file = "shuffled-full-set-hashed.csv"):
    # Import Data
    df = pd.read_csv(data_file, header=None, names=["Label", "Data"])

    # Drop rows with any empty cells
    df.dropna(axis=0, how='any', inplace=True)






    # Could use Undersampling or oversampling - https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis

    # Divide data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['Data'], df['Label'], random_state = 12)


    # Vectorization using bag of words model
    count_vect = CountVectorizer()
    vect = count_vect.fit(X_train)
    X_train_counts = count_vect.transform(X_train)



    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Save the vocabulary
    dictionary_filepath = "vocabulary.sav"
    pickle.dump(vect.vocabulary_, open(dictionary_filepath, 'wb'), protocol=2)


    # Train Model
    model = LinearSVC()
    clf = model.fit(X_train_tfidf, y_train)

    # Save Model
    model_filename = "trained_model.sav"
    pickle.dump(model, open(model_filename, 'wb'), protocol=2)


    # Test Model
    y_pred = model.predict(count_vect.transform(X_test))


    # Check Model
    conf_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(conf_mat)
    print("Accuracy_score = ", accuracy)
    return accuracy


train_model()