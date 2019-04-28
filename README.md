# TextClassifier
### Note all code assumes write privileges for the directory in which the code is run
### Code assumes tensorflow, keras, and scikit-learn are installed
#### tensorflow and keras can be installed via the following command:
`pip install -r requirements.txt`
#### sklearn should be installed as follows:
`pip install sklearn`

This solution applies two NLP techniques to classify text data. The first is Naive Bayes and the second is an LSTM Network.

Code is divided into .py files. Use of these files is shown in nlp_model_builder.ipynb

## To score a new dataset (containing the same fields, headers and classes as case_study_data.csv)
from the command line:
`python use_trained_nb_model.py new_dataset.csv`
`python use_trained_lstm_model.py new_dataset.csv`

## The above commands will generate tsv files (named nb_predictions.tsv and lstm_predictions.tsv)