import csv
import os
import pandas as pd
import shutil

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from vocabulary_processor_helpers import _text_normalizer

pd.options.mode.chained_assignment = None

def process_held_out_data(input_file):
    print("Processing Data. Please Wait!")
    data = pd.read_csv(input_file)
    data.loc[:, 'text'] = data['text'].apply(lambda x: _text_normalizer(x))
    y = data.pop(u'product_group')
    data.loc[:,'product_group'] = y
    data.to_csv('held_out_data.tsv', sep='\t',index=False, header=None)

def split_data(input_file):
    print("Processing Data. Please Wait!")
    if os.path.exists("data"):
        shutil.rmtree("data")
    os.makedirs("data")
    data = pd.read_csv(input_file)
   
    data.loc[:, 'text'] = data['text'].apply(lambda x: _text_normalizer(x))
   
    y = data.pop(u'product_group')
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
   
    X_train.loc[:,'product_group'] = y_train
    X_test.loc[:,'product_group'] = y_test
    X_val.loc[:,'product_group'] = y_val
    X_train = shuffle(X_train)

    X_train.to_csv(os.path.join('data', 'training_data.tsv'), sep='\t',index=False, header=None)
    X_test.to_csv(os.path.join('data','test_data.tsv'), sep='\t',index=False, header=None)
    X_val.to_csv(os.path.join('data','validation_data.tsv'), sep='\t',index=False, header=None)

    total_training_samples = X_train.shape[0]
    total_test_samples = X_test.shape[0]
    total_validation_samples = X_val.shape[0]

    label_mapping = dict(list((x[1], x[0]) for x in enumerate(sorted(X_train['product_group'].unique()))))
                         
    return label_mapping, total_training_samples, total_validation_samples, total_test_samples
