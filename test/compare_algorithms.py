from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import pandas as pd


data_file = "shuffled-full-set-hashed.csv"


# Import Data
df = pd.read_csv(data_file, header=None, names=["Label", "Data"])

# Drop rows with any empty cells
df.dropna(axis=0, how='any', inplace=True)

# Divide data into training and testing sets

X = df['Data']
Y = df['Label']
#X_train, X_test, y_train, y_test = train_test_split(df['Data'], df['Label'], random_state=0)
X_train = X
y_train = Y



# Vectorization using bag of words model
count_vect = CountVectorizer()
vect = count_vect.fit(X_train)
X_train_counts = count_vect.transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# prepare configuration for cross validation test harness
seed = 7

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# models.append(('RFC', RandomForestClassifier()))
# models.append(('LSVC', LinearSVC()))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train_tfidf, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


