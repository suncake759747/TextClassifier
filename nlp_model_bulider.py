import config
import math
import numpy as np
import os
import pickle
import sys

from keras_model_helpers import build_keras_model, use_lstm
from nb_model_helpers import build_nb_model, use_nb
from util import split_data
from vocabulary_processor_helpers import build_vocabulary_processor, _text_normalizer, _tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_file_name = sys.argv[1]
label_mapping, total_training_samples, total_validation_samples, total_test_samples = split_data(data_file_name)
train_steps = int(math.ceil(total_training_samples / config.batch_size))
validation_steps = int(math.ceil(total_validation_samples / config.batch_size))
with open('label_mapping_dict.p', 'wb') as f:
    pickle.dump(label_mapping, f)
reverse_label_mapping = dict((x[1], x[0]) for x in label_mapping.items())
with open('reverse_label_mapping_dict.p', 'wb') as f:
    pickle.dump(reverse_label_mapping, f)

data_root = 'data/'

validation_data_path = os.path.join(data_root, 'validation_data.tsv')
test_data_path = os.path.join(data_root, 'test_data.tsv')
training_data_path = os.path.join(data_root, 'training_data.tsv')

print("Building a Naive Bayes model")

nb_model = build_nb_model(training_data_path, test_data_path, validation_data_path)

nb_probs, true_classes, nb_preds, ids = use_nb(test_data_path)

print("Building an LSTM model")

vocab_processor = build_vocabulary_processor(training_data_path, config.max_len, config.min_word_count_freq)
with open('vocab_processor.p', 'wb') as f:
    pickle.dump(vocab_processor, f)
    
model = build_keras_model(config.batch_size, config.dropout_rate, config.embedding_size, config.max_len, 
                          config.num_epochs, train_steps, validation_steps, vocab_processor, label_mapping,
                          training_data_path, validation_data_path, model_name='saved_keras_model')

lstm_probs, true_classes, lstm_preds, ids = use_lstm(model, test_data_path, vocab_processor, label_mapping)