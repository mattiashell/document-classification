import pickle
from sklearn.feature_extraction.text import CountVectorizer

def predict(data):

    # Load trained model
    file_name = "trained_model.sav"
    trained_model = pickle.load(open(file_name, 'rb'))

    # Load vocabulary
    dictionary_filepath = "vocabulary.sav"
    loaded_vocabulary = pickle.load(open(dictionary_filepath, 'rb'))

    count_vect = CountVectorizer(vocabulary=loaded_vocabulary)
    count_vect._validate_vocabulary()

    return trained_model.predict(count_vect.transform([data]))[0]




    # print("Expected : ", df1["Label"].at[0])
    # print("Predicted : ", model.predict(count_vect.transform([df1["Data"].at[0]]))[0])
