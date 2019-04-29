import urllib.request
urllib.request.urlretrieve("https://www.dropbox.com/s/k66bxjwhdo70h3a/nb_classifier.pickle?dl=1", "nb_classifier.pickle")
urllib.request.urlretrieve("https://www.dropbox.com/s/68bpdh60muuykh2/nb_labelencoder.pickle?dl=1", "nb_labelencoder.pickle")
urllib.request.urlretrieve("https://www.dropbox.com/s/afz4x5u1iq2mlo0/nb_vectorizer.pickle?dl=1", "nb_vectorizer.pickle")
print("NB Models Downloaded")