import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pickle

def build_nb_model(training_data_path, testing_data_path, validation_data_path):
    
    trainData = pd.read_csv(training_data_path, sep='\t', header=None, names = ['complaint_id', u'text', u'product_group'])
    testData = pd.read_csv(testing_data_path, sep='\t', header=None, names =['complaint_id', u'text', u'product_group'])
    validationData = pd.read_csv(validation_data_path, sep='\t', header=None, names =['complaint_id', u'text', u'product_group'])

    y_train_df = trainData.pop(u'product_group')
    X_train_df = trainData 
    y_val_df = validationData.pop(u'product_group')
    X_val_df = validationData
    y_test_df = testData.pop(u'product_group')
    X_test_df = testData

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train_df)
    pickle.dump(le, open("nb_labelencoder.pickle", "wb"))

    count_vectorizer = CountVectorizer(ngram_range = (1,3), min_df = 3, stop_words='english')
    print("Processing Training Data")
    X_train = count_vectorizer.fit_transform(X_train_df['text'])
    pickle.dump(count_vectorizer, open("nb_vectorizer.pickle", "wb"))

    print("Processing Validation Data")
    X_val = count_vectorizer.transform(X_val_df['text'])
    y_val = le.transform(y_val_df)

    print("Processing Test Data")
    X_test = count_vectorizer.transform(X_test_df['text'])
    y_test = le.transform(y_test_df)

    nb_classifier = MultinomialNB(alpha=0.0001)
    nb_classifier.fit(X_train, y_train)
    predictions = nb_classifier.predict(X_val)
    prediction_probabilities = nb_classifier.predict_proba(X_val)[:,1]

    print("Confusion Matrix for VALIDATION data in %s" % validation_data_path)
    print("PREDICTED CLASS ON X-AXIS. TRUE CLASS ON Y-AXIS.")    
    confusion_df = pd.DataFrame(data = confusion_matrix(y_val ,predictions), columns = le.classes_, index=le.classes_)
    print(confusion_df)
    print(classification_report(y_val ,predictions,labels=range(7), target_names=le.classes_))

    pickle.dump(nb_classifier, open("nb_classifier.pickle", 'wb'))

    return nb_classifier

def use_nb(test_data_path):
    testData = pd.read_csv(test_data_path, sep='\t', header=None, names =['complaint_id', u'text', u'product_group'])
    
    y_test_df = testData.pop(u'product_group')
    X_test_df = testData

    le = pickle.load(open('nb_labelencoder.pickle', 'rb'))
    count_vectorizer = pickle.load(open('nb_vectorizer.pickle', 'rb'))
    nb_classifier = pickle.load(open('nb_classifier.pickle', 'rb'))

    X_test = count_vectorizer.transform(X_test_df['text'])
    y_test = le.transform(y_test_df)

    predictions = nb_classifier.predict(X_test)

    prediction_probabilities = nb_classifier.predict_proba(X_test)[:,1]

    confusion_df = pd.DataFrame(data = confusion_matrix(y_test,predictions), columns = le.classes_, index=le.classes_)
    print("Confusion Matrix for TEST data in %s" % test_data_path)
    print("PREDICTED CLASS ON X-AXIS. TRUE CLASS ON Y-AXIS.")
    print(confusion_df)
    print(classification_report(y_test,predictions,labels=list(range(7)), target_names=le.classes_))

    ids = testData.pop(u'complaint_id')
    predictions_labels = le.inverse_transform(predictions)

    return prediction_probabilities, y_test_df, predictions_labels, ids






