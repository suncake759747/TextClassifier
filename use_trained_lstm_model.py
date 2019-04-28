import os
import pickle
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
from keras_model_helpers import use_lstm
from tensorflow.contrib import learn
from util import process_held_out_data
from vocabulary_processor_helpers import build_vocabulary_processor, _text_normalizer, _tokenizer

held_out_file = sys.argv[1]
process_held_out_data(held_out_file)

with open('label_mapping_dict.p', 'rb') as f:
    label_mapping = pickle.load(f)
with open('reverse_label_mapping_dict.p', 'rb') as f:
    reverse_label_mapping = pickle.load(f)
with open('vocab_processor.p', 'rb') as f:
    vocab_processor = pickle.load(f)
    
presentation_model = load_model('saved_keras_model')
    
lstm_probs, true_classes, lstm_preds, ids = use_lstm(presentation_model, 'held_out_data.tsv', vocab_processor, label_mapping)

with open ('lstm_predictions.tsv', 'w') as f:
    for index in range(ids.shape[0]):
        f.write("%s\t%s\n" % (ids[index,0], reverse_label_mapping[lstm_preds[index,0]]))